"""Hosted-notebook input discovery and extraction; never trains or selects rules."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

from .experiment import digest, object_hash
from .rule_suite import validation_context

REQUIRED = ('experiment/manifest.json', 'experiment/selection.json', 'checkpoints/ensemble.pt',
            'experiment/predictions/B.npz', 'experiment/predictions/C.npz',
            'experiment/predictions/test.npz')


def zip_prefix(path, *, require_test=True):
    """Accept one complete run at ZIP root or inside one enclosing directory."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = REQUIRED if require_test else REQUIRED[:-1]
            prefixes = [name[:-len(required[0])] for name in names if name.endswith(required[0])]
            prefixes = [prefix for prefix in prefixes if all(prefix + name in names for name in required)]
            if len(prefixes) != 1:
                return None
            return prefixes[0]
    except (OSError, zipfile.BadZipFile):
        return None


def find_sources(roots, *, require_test=True):
    """Search bounded input roots, deduplicating ZIP/extracted copies by manifest."""
    found = {}
    required = REQUIRED if require_test else REQUIRED[:-1]
    for root in dict.fromkeys(Path(p).expanduser().resolve() for p in roots):
        if not root.exists():
            continue
        # Callers supply input directories, not entire package/checkpoint trees.
        manifests = [root / REQUIRED[0]] if (root / REQUIRED[0]).is_file() else root.rglob('manifest.json')
        for manifest in manifests:
            source = manifest.parent.parent
            if manifest.parent.name == 'experiment' and all((source / p).is_file() for p in required):
                found.setdefault(digest(manifest), source)
        archives = [root] if root.is_file() else root.rglob('*.zip')
        for path in archives:
            prefix = zip_prefix(path, require_test=require_test)
            if prefix is not None:
                with zipfile.ZipFile(path) as archive:
                    identity = hashlib.sha256(archive.read(prefix + REQUIRED[0])).hexdigest()
                found.setdefault(identity, path)
    return sorted(found.values(), key=str)


def extract_source(path, cache_dir, *, require_test=True):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        sources = find_sources([path], require_test=require_test)
        if len(sources) != 1:
            raise ValueError('SOURCE directory must contain exactly one complete H2 run; '
                             f'found {len(sources)}. Point SOURCE at the desired run or ZIP.')
        if sources[0].is_dir():
            return sources[0]
        return extract_source(sources[0], cache_dir, require_test=require_test)
    prefix = zip_prefix(path, require_test=require_test)
    if prefix is None:
        raise ValueError('SOURCE must be a complete H2 results ZIP or extracted run, '
                         'including the ensemble and B/C/test prediction caches.')
    identity = digest(path)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / identity
    if destination.exists():
        source = destination / prefix
        if not all((source / name).is_file() for name in (REQUIRED if require_test else REQUIRED[:-1])):
            raise ValueError('Cached extraction is incomplete; remove only that extraction directory and retry.')
        return source
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
        # Reject path escapes, symbolic links and duplicate members before writing.
        names = [member.filename for member in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError('Archive contains duplicate members')
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.filename or
                    stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Archive contains an unsafe path or symbolic link')
        archive.extractall(temporary)
        os.replace(temporary, destination)
    return destination / prefix


def prepare_inputs(source, roots, work_dir, config_path, suite_name=None, upload=None):
    """One input choice; auto-named output changes when data, code or grid changes."""
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if source is None:
        choices = find_sources([*roots, work_dir / 'source-inputs'])
        if not choices and upload is not None:
            choices = find_sources(upload())
        if len(choices) != 1:
            listing = '\n'.join(str(p) for p in choices)
            raise ValueError('Attach/upload one original H2 results ZIP, or set SOURCE to its path. '
                             f'Found {len(choices)} candidate runs.\n{listing}')
        source = choices[0]
    source = extract_source(source, work_dir / 'source-inputs')
    # This validates B/C identities but never opens the test prediction array.
    _, _, manifest, _, _, identity = validation_context(source, config_path)
    if suite_name is None:
        provenance = manifest['provenance']
        suite_name = f"h2-{provenance['track']}-seed{provenance['seed']}-rules-{object_hash(identity)[:10]}"
    if not isinstance(suite_name, str) or not suite_name or Path(suite_name).name != suite_name or suite_name in ('.', '..') or '\\' in suite_name:
        raise ValueError('SUITE_NAME must be one directory name')
    output = work_dir / 'results' / suite_name
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output):
        pass
    return source, output
