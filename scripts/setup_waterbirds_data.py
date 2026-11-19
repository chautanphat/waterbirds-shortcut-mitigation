"""Download the pinned HF Waterbirds snapshot and audit canonical metadata joins."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_REVISION='0b3b8b49a57aecf49b6e9ea707d9f120535c20d5'
PARQUET_REVISION='110dcb098399c0fc5928e9f9ce52e0fa19b1b4f7'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'data/waterbirds')
    p.add_argument('--revision', default=SOURCE_REVISION)
    p.add_argument('--parquet-revision',default=PARQUET_REVISION)
    p.add_argument('--skip-download', action='store_true')
    p.add_argument('--force-download',action='store_true',help='Bypass the materialized snapshot hash cache')
    p.add_argument('--prune-download-cache',action='store_true',help='Remove redundant bundled Parquet files after hash-audited materialization')
    p.add_argument('--cache-only',action='store_true',help='With --skip-download, maintain snapshot hashes/cache without rebuilding already-audited manifests')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--representation', choices=['parquet', 'files'], default='parquet')
    p.add_argument('--cub-root', type=Path)
    p.add_argument('--download-cub', action='store_true', help='Fetch original CUB RGB and official masks for verified cutouts')
    args = p.parse_args()
    if args.cache_only and not args.skip_download:
        p.error('--cache-only requires --skip-download')
    if args.representation=='parquet' and args.revision!=SOURCE_REVISION and args.parquet_revision==PARQUET_REVISION:
        p.error('A custom source revision needs its matching --parquet-revision, or use --representation files')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(root / '.hf_cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import HfApi, snapshot_download
    from src.counterfactual.common import sha256,write_json,contained
    inventory_path=root/'hf_snapshot_sha256.json'
    cached=False
    if not args.force_download and inventory_path.exists() and (root/'download_manifest.json').exists():
        inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
        existing=json.loads((root/'download_manifest.json').read_text(encoding='utf-8'))
        if inventory.get('revision')==args.revision==existing.get('revision') and set(inventory['files'])==set(existing['files']):
            cached=all(contained(root/'hf',path).is_file() and sha256(contained(root/'hf',path))==digest
                       for path,digest in inventory['files'].items())
            if cached:
                print('Pinned materialized HF snapshot verified; no network download needed',flush=True)
    if args.skip_download and inventory_path.exists() and not cached:
        raise ValueError('Materialized HF snapshot differs from its saved hashes; repair from the pinned source before offline validation')
    if not args.skip_download and not cached:
        info = HfApi().dataset_info('arubique/waterbirds', revision=args.revision)
        record = dict(repo_id='arubique/waterbirds', revision=info.sha,
                      files=[s.rfilename for s in info.siblings])
        (root / 'download_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        if args.representation == 'files':
            snapshot_download(repo_id=record['repo_id'], repo_type='dataset', revision=info.sha,
                              local_dir=str(root / 'hf'), max_workers=args.workers,force_download=args.force_download)
        else:
            # Official auto-conversion bundles 17,382 images into twelve files.
            # Fewer Hub requests than the image-folder snapshot; exact bytes retained.
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
            converted = HfApi().dataset_info('arubique/waterbirds', revision=args.parquet_revision)
            record['parquet_revision'] = converted.sha
            write_path = root / 'download_manifest.json'
            write_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            for filename in ['metadata.csv', 'README.md', '.gitattributes']:
                hf_hub_download(repo_id=record['repo_id'], repo_type='dataset', revision=info.sha,
                                filename=filename, local_dir=str(root / 'hf'),force_download=args.force_download)
            for sibling in converted.siblings:
                if not sibling.rfilename.endswith('.parquet'):
                    continue
                config = sibling.rfilename.split('/')[0]
                file = hf_hub_download(repo_id=record['repo_id'], repo_type='dataset', revision=converted.sha,
                                      filename=sibling.rfilename, local_dir=str(root / 'parquet'),force_download=args.force_download)
                table = pq.read_table(file)
                if table.column_names != ['image']:
                    raise ValueError(f'Unexpected Parquet schema: {table.schema}')
                for batch in table.to_batches(max_chunksize=128):
                    for image in batch.to_pydict()['image']:
                        path = image['path'].replace('\\', '/')
                        # Auto-converted paths are the original imagefolder-relative paths.
                        if '/' not in path:
                            group = config.removesuffix('_fg_only').removesuffix('_bg_only')
                            digit = '0' if group.startswith('landbird') else '1'
                            path = f'{config}/{digit}/{path}'
                        elif not path.startswith(config + '/'):
                            path = config + '/' + path
                        from src.counterfactual.common import contained
                        target = contained(root / 'hf', path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        blob = image['bytes']
                        if blob is None:
                            raise ValueError('Parquet conversion contains external images; use --representation files')
                        if target.exists() and target.read_bytes() != blob and not args.force_download:
                            raise ValueError(f'Parquet bytes disagree with original snapshot: {path}')
                        if not target.exists() or args.force_download:
                            target.write_bytes(blob)
                print(f'Materialized {config}: {len(table)} images', flush=True)
    record = json.loads((root / 'download_manifest.json').read_text(encoding='utf-8'))
    if not inventory_path.exists() or not cached:
        write_json(inventory_path,dict(revision=record['revision'],files={p:sha256(contained(root/'hf',p)) for p in record['files']}))
    if args.prune_download_cache:
        # Only redundant files created by this downloader, below its cache root.
        cache_root=(root/'parquet').resolve()
        removed=0
        for file in cache_root.rglob('*.parquet'):
            resolved=file.resolve()
            if not resolved.is_relative_to(cache_root) or not cache_root.is_relative_to(root):
                raise ValueError('Unexpected cache path; refusing to prune')
            resolved.unlink()
            removed+=1
        print(f'Pruned {removed} redundant Parquet bundles; materialized image hashes retained',flush=True)
    if args.cache_only:
        print('Snapshot/cache maintenance complete; existing dataset audit was preserved',flush=True)
        return
    from src.counterfactual.data import prepare_dataset, download_cub
    cub_root = args.cub_root
    if args.download_cub:
        cub_root = download_cub(root)
    report = prepare_dataset(root, record, cub_root=cub_root)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
