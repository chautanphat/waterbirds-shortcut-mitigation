"""Canonical HF joins, decoded-image audits, cutouts and background repositories."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import urllib.error
import requests

import numpy as np
from PIL import Image, ImageFilter

from .common import (LABELS, SPLITS, contained, pixel_hash, read_csv, sha256,
                     write_csv, write_json)

LICENSE = 'Noncommercial research/education; underlying Places/CUB image rights remain with owners'
GROUP_NAMES = ['landbird_on_land', 'landbird_on_water', 'waterbird_on_land', 'waterbird_on_water']


def download_cub(root):
    """Stream official archives, retaining only HF birds to bound disk usage."""
    root = Path(root)
    cub = root / 'cub'
    cub.mkdir(exist_ok=True)
    metadata = root / 'metadata.csv'
    wanted = {r['img_filename'] for r in read_csv(metadata) if r['split'] == '2'}
    specs = [('https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz',
              'segmentations.tgz', '4d47ba1228eae64f2fa547c47bc65255'),
             ('https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz',
              'CUB_200_2011.tgz', '97eceeb196236b17998738112f37df78')]
    for url, name, checksum in specs:
        marker = cub / (name + '.complete.json')
        if marker.exists():
            continue
        print(f'Streaming original CUB asset: {url}', flush=True)
        digest, full_digest = hashlib.md5(), hashlib.sha256()
        class CheckedStream:
            def __init__(self, stream):
                self.stream = stream
            def read(self, n=-1):
                block = self.stream.read(n)
                digest.update(block)
                full_digest.update(block)
                return block
        try:
            response = requests.get(url, stream=True, timeout=120)
        except requests.exceptions.SSLError:
            # Caltech's public HTTP record redirects to a valid signed HTTPS OSN
            # object. Do not disable TLS checks. Verify the official archive MD5.
            response = requests.get(url.replace('https://data.caltech.edu/', 'http://data.caltech.edu/'), stream=True, timeout=120)
            if not response.url.startswith('https://'):
                response.close()
                raise ValueError('Caltech fallback did not redirect to verified HTTPS')
        response.raise_for_status()
        with response:
          ranged = None
          if name == 'CUB_200_2011.tgz' and int(response.headers.get('Content-Length','0')) > 0:
              from .downloads import RangeStream
              ranged = RangeStream(url.replace('https://data.caltech.edu/', 'http://data.caltech.edu/'),
                                   response.url,int(response.headers['Content-Length']))
              response.close()
          stream = CheckedStream(ranged or response.raw)
          with tarfile.open(fileobj=stream, mode='r|gz') as tar:
            for m in tar:
                if not m.isfile() or not m.name.lower().endswith(('.jpg', '.png', '.txt')):
                    continue
                canonical = '/'.join(m.name.split('/')[-2:])
                if canonical.replace('.png', '.jpg') not in wanted:
                    continue
                target = contained(cub, m.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, target.open('wb') as out:
                    shutil.copyfileobj(src, out)
          while stream.read(1024 * 1024):
              pass
          if ranged:
              ranged.close()
        if digest.hexdigest() != checksum:
            raise ValueError(f'CUB archive checksum mismatch: {url}')
        write_json(marker, dict(url=url, md5=checksum, sha256=full_digest.hexdigest(), selected_split='test'))
    return cub


def metadata_index(rows):
    index, ids = {}, set()
    for r in rows:
        if not str(r['img_id']).isdigit():
            raise ValueError('Canonical image IDs must be nonnegative integers')
        if (r['y'] not in ('0', '1') or r['place'] not in ('0', '1')
                or r['split'] not in ('0', '1', '2')):
            raise ValueError('Invalid canonical label, place or split')
        key = r['img_filename'].replace('/', '_') + '.jpg'
        if key in index or r['img_id'] in ids:
            raise ValueError('Duplicate canonical image identity')
        contained(Path('.'), r['img_filename'])
        index[key] = r
        ids.add(r['img_id'])
    return index


def image_audit(path):
    with Image.open(path) as im:
        im.load()
        rgb = im.convert('RGB')
        # Average hash is a candidate near-duplicate screen, not semantic similarity.
        small = np.asarray(rgb.resize((8, 8)).convert('L'))
        ahash = np.packbits(small >= small.mean()).tobytes().hex()
        return dict(sha256=sha256(path), pixel_sha256=pixel_hash(rgb),
                    width=rgb.width, height=rgb.height, average_hash=ahash)


def prepare_dataset(root, record, *, cub_root=None):
    root = Path(root).resolve()
    hf = root / 'hf'
    metadata = hf / 'metadata.csv'
    rows = read_csv(metadata)
    index = metadata_index(rows)
    if (root / 'metadata.csv').exists() and sha256(root / 'metadata.csv') != sha256(metadata):
        raise ValueError('Existing canonical metadata differs from HF; originals were preserved')
    if not (root / 'metadata.csv').exists():
        shutil.copyfile(metadata, root / 'metadata.csv')
    files = sorted(record['files'])
    scenes = [p for p in files if p.split('/')[0] in GROUP_NAMES and p.endswith('.jpg')]
    foregrounds, backgrounds, invalid, hashes = [], [], [], defaultdict(list)
    prior_file=root/'foregrounds_manifest.csv'
    prior={r['foreground_id']:r for r in read_csv(prior_file)} if prior_file.exists() else {}
    (root / 'foregrounds/masks').mkdir(parents=True, exist_ok=True)
    (root / 'foregrounds/rgba').mkdir(parents=True, exist_ok=True)
    if cub_root is None and all((root/'cub'/(name+'.complete.json')).exists()
                               for name in ('segmentations.tgz','CUB_200_2011.tgz')):
        cub_root = root / 'cub'
    for n, source in enumerate(scenes):
        row = index.get(Path(source).name)
        if row is None:
            raise ValueError(f'Cannot join HF file to canonical metadata: {source}')
        y, a = int(row['y']), int(row['place'])
        expected = GROUP_NAMES[2 * y + a]
        if source.split('/')[0] != expected:
            raise ValueError(f'Canonical group disagrees with HF scenario: {source}')
        # Folder digits are audited separately; they are NEVER a label source.
        if source.split('/')[1] != str(y):
            raise ValueError(f'Unexpected folder encoding; inspect dataset revision: {source}')
        fg_source = source.replace(expected + '/', expected + '_fg_only/')
        bg_source = source.replace(expected + '/', expected + '_bg_only/')
        try:
            audits = [image_audit(contained(hf, p)) for p in (source, fg_source, bg_source)]
            with Image.open(hf / source) as im:
                size = im.size
            with Image.open(hf / fg_source) as im:
                if im.size != size:
                    raise ValueError('Foreground-only geometry differs from source scene')
        except (OSError, ValueError) as error:
            invalid.append(dict(source=source, reason=str(error)))
            continue
        split = SPLITS[int(row['split'])]
        bird_id = row['img_id']
        background_id = 'hf-bg-' + bird_id
        bg_audit = audits[2]
        place = row['place_filename']
        scene_category = '/'.join(place.split('/')[2:-1])
        if scene_category not in (['bamboo_forest','forest/broadleaf'] if a==0 else ['ocean','lake/natural']):
            raise ValueError(f'Canonical Places category disagrees with place label: {place}')
        b = dict(background_id=background_id, bg_type=['land', 'water'][a],
                 scene_category=scene_category, source_dataset=record['repo_id'],
                 source_path='hf/' + bg_source, original_split=split,
                 canonical_bird_id=bird_id, place_filename=place,
                 source_revision=record['revision'], license_note=LICENSE,
                 content_filter_status='not_screened_for_humans_or_animals', **bg_audit)
        backgrounds.append(b)
        hashes[bg_audit['pixel_sha256']].append(background_id)
        mask_file = root / 'foregrounds/masks' / (bird_id + '.png')
        rgba_file = root / 'foregrounds/rgba' / (bird_id + '.png')
        original_rgb = mask = None
        if cub_root:
            cub = Path(cub_root)
            original_rgb = next((p for p in [cub / 'CUB_200_2011/images' / row['img_filename'],
                                             cub / 'images' / row['img_filename']] if p.is_file()), None)
            mask = next((p for p in [cub / 'segmentations' / Path(row['img_filename']).with_suffix('.png'),
                                     cub / 'CUB_200_2011/segmentations' / Path(row['img_filename']).with_suffix('.png')]
                         if p.is_file()), None)
        quality = 'verified_original_cub' if original_rgb and mask else 'approximate_gray_matte'
        if cub_root and quality != 'verified_original_cub':
            raise FileNotFoundError(f'Original RGB or segmentation missing for {row["img_filename"]} under --cub-root; provide a complete matching CUB root')
        version = 'cub-official-v1' if quality == 'verified_original_cub' else 'gray150-threshold12-erode1-v1'
        rgb_hash=sha256(original_rgb) if original_rgb else ''
        original_mask_hash=sha256(mask) if mask else ''
        old=prior.get('hf-fg-'+bird_id,{})
        unchanged=(old.get('quality_status')==quality and old.get('mask_version')==version
            and old.get('original_cub_sha256','')==rgb_hash and old.get('original_mask_sha256','')==original_mask_hash
            and rgba_file.exists() and mask_file.exists()
            and old.get('rgba_sha256')==sha256(rgba_file) and old.get('mask_sha256')==sha256(mask_file))
        if not unchanged:
            with Image.open(original_rgb or hf / fg_source) as im:
                rgb = im.convert('RGB')
            if mask and original_rgb:
                with Image.open(mask) as im:
                    alpha = im.convert('L')
                if rgb.size != size or alpha.size != size:
                    raise ValueError(f'Original CUB geometry differs for bird {bird_id}')
            else:
                # The derivative is a lossy gray matte, NOT a transparent cutout.
                # This extraction is explicitly approximate and excluded from controlled claims.
                array = np.asarray(rgb).astype(np.int16)
                support = np.max(np.abs(array - 150), axis=2) > 12
                alpha = Image.fromarray((support * 255).astype('uint8')).filter(ImageFilter.MinFilter(3))
            if not alpha.getbbox() or np.asarray(alpha).min() == 255:
                raise ValueError(f'Missing/invalid segmentation for bird {bird_id}')
            alpha.save(mask_file)
            pixels = np.array(rgb)
            pixels[np.asarray(alpha) == 0] = 0
            rgb = Image.fromarray(pixels)
            rgb.putalpha(alpha)
            rgb.save(rgba_file)
        foregrounds.append(dict(foreground_id='hf-fg-' + bird_id, bird_source_id=bird_id,
            species=row['img_filename'].split('/')[0], y=y, original_group=2*y+a,
            original_split=split, source_image_path='hf/' + source,
            canonical_filename=row['img_filename'], foreground_source_path='hf/' + fg_source,
            rgba_path=rgba_file.relative_to(root).as_posix(), mask_path=mask_file.relative_to(root).as_posix(),
            source_sha256=audits[0]['sha256'], original_rgb_path=str(original_rgb) if original_rgb else '',
            original_mask_path=str(mask) if mask else '',original_cub_sha256=rgb_hash,
            original_mask_sha256=original_mask_hash,
            rgba_sha256=sha256(rgba_file), mask_sha256=sha256(mask_file),
            mask_version=version, quality_status=quality, source_revision=record['revision']))
        if (n + 1) % 500 == 0:
            print(f'Audited {n+1}/{len(scenes)} paired assets', flush=True)
    write_csv(root / 'backgrounds_manifest.csv', backgrounds)
    write_csv(root / 'foregrounds_manifest.csv', foregrounds)
    write_json(root / 'quarantine.json', invalid)
    duplicates = [v for v in hashes.values() if len(v) > 1]
    near = defaultdict(list)
    for b in backgrounds:
        near[b['average_hash']].append(b['background_id'])
    near_candidates = [v for v in near.values() if len(v) > 1]
    write_json(root / 'duplicate_audit.json', dict(exact_pixel_duplicates=duplicates,
        near_duplicate_candidates=near_candidates, near_duplicate_method='8x8 average-hash collision; inspect manually'))
    if not foregrounds or {b['bg_type'] for b in backgrounds} != {'land', 'water'}:
        raise ValueError('Dataset lacks usable birds or both background categories')
    place_splits=defaultdict(set)
    for r in rows:
        place_splits[r['place_filename']].add(SPLITS[int(r['split'])])
    cross_split_places={p:sorted(s) for p,s in place_splits.items() if len(s)>1}
    report = dict(status='validated' if not invalid else 'quarantined_invalid_files',
        source=record['repo_id'], revision=record['revision'], parquet_revision=record.get('parquet_revision'),expected_files=len(files),
        local_files=sum((hf / p).is_file() for p in files), metadata_sha256=sha256(metadata),
        canonical_metadata_rows=len(rows), canonical_metadata_split_counts=dict(Counter(SPLITS[int(r['split'])] for r in rows)),
        downloaded_official_split_counts=dict(Counter(f['original_split'] for f in foregrounds)),
        foreground_count=len(foregrounds), backgrounds=dict(Counter(b['bg_type'] for b in backgrounds)),
        mask_quality_counts=dict(Counter(f['quality_status'] for f in foregrounds)),
        metadata_join_coverage=len(foregrounds)/len(scenes), invalid_files=len(invalid),
        duplicate_background_sets=len(duplicates), near_duplicate_candidate_sets=len(near_candidates),
        cross_split_places_filename_conflicts=cross_split_places,
        scene_categories=dict(Counter(b['scene_category'] for b in backgrounds)),
        folder_digit_encoding='Observed 0=landbird,1=waterbird; conflicts with card; canonical metadata is authoritative',
        leakage_audit='All supplied HF images are official test. No train/val images provided; test use is retrospective_exploratory.',
        license_note=LICENSE)
    write_json(root / 'dataset_audit.json', report)
    return report


class BackgroundRepository:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = read_csv(self.root / 'backgrounds_manifest.csv')
        local = self.root / 'local_backgrounds_manifest.csv'
        if local.exists():
            self.rows += read_csv(local)
        self.by_id = {r['background_id']: r for r in self.rows}
        if len(self.rows) != len(self.by_id):
            raise ValueError('Duplicate background IDs')

    def list(self, category=None, split=None, source=None):
        return [r for r in self.rows if (category is None or r['bg_type'] == category)
                and (split is None or r['original_split'] == split)
                and (source is None or r['source_dataset'] == source)]

    def get(self, background_id):
        r = self.by_id[background_id]
        path = contained(self.root, r['source_path'])
        if sha256(path) != r['sha256']:
            raise ValueError('Background changed since audit; rerun dataset setup')
        with Image.open(path) as im:
            return im.convert('RGB'), dict(r)


class ForegroundRepository:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = read_csv(self.root / 'foregrounds_manifest.csv')
        external = self.root / 'external_foregrounds_manifest.csv'
        if external.exists():
            self.rows += read_csv(external)
        self.by_id = {r['foreground_id']: r for r in self.rows}
        if len(self.rows) != len(self.by_id):
            raise ValueError('Duplicate foreground IDs')

    def get(self, foreground_id):
        r = dict(self.by_id[foreground_id])
        for key, hash_key in [('rgba_path', 'rgba_sha256'), ('mask_path', 'mask_sha256')]:
            if sha256(contained(self.root, r[key])) != r[hash_key]:
                raise ValueError('Foreground/mask changed since audit; save a new mask version')
        if sha256(contained(self.root, r['source_image_path'])) != r['source_sha256']:
            raise ValueError('Canonical source image changed since audit')
        with Image.open(contained(self.root, r['rgba_path'])) as im:
            return im.convert('RGBA'), r


def import_backgrounds(root, manifest):
    """Explicit local Places manifest: path,bg_type,scene_category,original_split."""
    root, manifest = Path(root), Path(manifest).resolve()
    result, quarantine = [], []
    for r in read_csv(manifest):
        if r['bg_type'] not in ('land', 'water') or r['original_split'] not in ('train', 'val', 'test', 'unknown'):
            raise ValueError('Local backgrounds need explicit type and source split')
        src = (manifest.parent / r['path']).resolve()
        try:
            audit = image_audit(src)
        except OSError as error:
            quarantine.append(dict(path=str(src), error=str(error)))
            continue
        dest = root / 'local_backgrounds' / (audit['sha256'] + src.suffix.lower())
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copyfile(src, dest)
        result.append(dict(background_id='local-' + audit['sha256'][:20], bg_type=r['bg_type'],
            scene_category=r['scene_category'], original_split=r['original_split'], source_dataset='local_places_import',
            source_path=dest.relative_to(root).as_posix(), import_origin=str(src),
            license_note=r.get('license_note', LICENSE), content_filter_status='not_screened', **audit))
    # Explicit importer is idempotent and deduplicates byte-identical files.
    previous = root / 'local_backgrounds_manifest.csv'
    merged = {r['background_id']: r for r in (read_csv(previous) if previous.exists() else [])}
    merged.update({r['background_id']: r for r in result})
    write_csv(previous, merged.values())
    write_json(root / 'local_background_quarantine.json', quarantine)
    return dict(imported=len(result), unique=len(merged), quarantined=len(quarantine))
