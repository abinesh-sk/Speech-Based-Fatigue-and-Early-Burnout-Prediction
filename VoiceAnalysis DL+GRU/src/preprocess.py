from pathlib import Path
from typing import Generator, Iterable, Tuple

from .config import RAW_DATA_DIR

AudioPath = Path


def iter_audio_files(root: Path, extensions: Iterable[str] = (".wav", ".flac", ".mp3")) -> Generator[AudioPath, None, None]:
    """Yield audio file paths under root lazily."""
    ext_set = {ext.lower() for ext in extensions}
    yield from (
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ext_set
    )


def dataset_audio_paths(dataset_name: str, extensions: Iterable[str] = (".wav", ".flac", ".mp3")) -> Generator[AudioPath, None, None]:
    """Yield audio files for a named dataset (under data/raw/<dataset_name>/)."""
    dataset_root = RAW_DATA_DIR / dataset_name
    if dataset_root.exists():
        yield from iter_audio_files(dataset_root, extensions=extensions)


def iter_dataset_and_file(extensions: Iterable[str] = (".wav", ".flac", ".mp3")) -> Generator[Tuple[str, AudioPath], None, None]:
    """Yield (dataset_name, file_path) for all datasets present in data/raw."""
    ext_set = {ext.lower() for ext in extensions}
    yield from (
        (dataset_dir.name, path)
        for dataset_dir in RAW_DATA_DIR.iterdir()
        if dataset_dir.is_dir()
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ext_set
    )


__all__ = [
    "iter_audio_files",
    "dataset_audio_paths",
    "iter_dataset_and_file",
]
