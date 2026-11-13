"""Package readable local source for a fresh Kaggle notebook without GitHub writes."""
from pathlib import Path
import zipfile


def build(destination):
    root = Path(__file__).resolve().parents[3]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [*root.glob("src/**/*.py"), root/"configs/dpe_independent/study.yaml",
             root/"src/methods/dpe_independent/README.md", root/"src/methods/dpe_independent/requirements.txt"]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root))
    return destination
