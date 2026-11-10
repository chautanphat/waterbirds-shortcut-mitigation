"""Hosted input discovery/extraction; fixtures never use real model accuracy."""
import zipfile

import pytest

from src.methods.dpe_head_reweighting.notebook_setup import (
    REQUIRED, extract_source, find_sources, prepare_inputs,
)


def bundle(tmp_path, name='one', manifest=b'{"fixture": "one"}'):
    source = tmp_path / name
    for filename in REQUIRED:
        path = source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(manifest if filename == REQUIRED[0] else b'fixture')
    archive = tmp_path / f'{name}.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for filename in REQUIRED:
            zipped.write(source / filename, 'outer/run/' + filename)
    return source, archive


def test_discover_deduplicates_and_extracts_nested_readonly_zip(tmp_path):
    source, archive = bundle(tmp_path)
    assert find_sources([source, archive]) == [source]
    archive.chmod(0o444)
    before = archive.read_bytes()
    extracted = extract_source(archive, tmp_path / 'cache')
    assert all((extracted / name).is_file() for name in REQUIRED)
    assert extracted == extract_source(archive, tmp_path / 'cache')
    assert archive.read_bytes() == before
    assert source == extract_source(source, tmp_path / 'cache')


def test_rejects_unsafe_archive_paths_and_incomplete_bundles(tmp_path):
    _, archive = bundle(tmp_path)
    with zipfile.ZipFile(archive, 'a') as zipped:
        zipped.writestr('../escape.txt', 'bad')
    with pytest.raises(ValueError, match='unsafe'):
        extract_source(archive, tmp_path / 'cache')
    assert not (tmp_path / 'escape.txt').exists()
    incomplete = tmp_path / 'incomplete.zip'
    with zipfile.ZipFile(incomplete, 'w') as zipped:
        zipped.writestr(REQUIRED[0], '{}')
    with pytest.raises(ValueError, match='complete source ZIP'):
        extract_source(incomplete, tmp_path / 'cache')


def test_ambiguous_source_lists_choices_without_guessing(tmp_path):
    one, _ = bundle(tmp_path, 'one')
    two, _ = bundle(tmp_path, 'two', b'{"fixture": "two"}')
    with pytest.raises(ValueError, match='Found 2 candidate runs') as error:
        prepare_inputs(None, [one, two], tmp_path / 'work', tmp_path / 'unused-config')
    assert str(one) in str(error.value) and str(two) in str(error.value)
