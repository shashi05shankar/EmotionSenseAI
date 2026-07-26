"""Fetch real corpora into data/raw/<name>/ for RC1 experimental validation.

Prepared for the validation milestone (no source-code changes). Lands each corpus in the
layout the frozen loaders expect (they rglob for *.wav, so any internal folder structure is
fine):

  data/raw/ravdess/   RAVDESS speech (Zenodo, public, no auth)
  data/raw/tess/      TESS       (Kaggle: ejlok1/toronto-emotional-speech-set-tess)
  data/raw/crema_d/   CREMA-D    (Kaggle: ejlok1/cremad)

RAVDESS downloads directly from Zenodo with no credentials. TESS and CREMA-D use the Kaggle
CLI (needs a configured ~/.kaggle/kaggle.json); if Kaggle is unavailable the script prints
manual instructions instead of failing hard.

Usage:
    python scripts/fetch_datasets.py --all
    python scripts/fetch_datasets.py ravdess
    python scripts/fetch_datasets.py tess crema_d
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
KAGGLE_SLUGS = {
    "tess": "ejlok1/toronto-emotional-speech-set-tess",
    "crema_d": "ejlok1/cremad",
}


def _has_wavs(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.wav"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")

    def _hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def fetch_ravdess() -> None:
    out = RAW / "ravdess"
    if _has_wavs(out):
        print("ravdess: already present, skipping")
        return
    print("ravdess: fetching from Zenodo...")
    zip_path = RAW / "ravdess.zip"
    _download(RAVDESS_URL, zip_path)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    zip_path.unlink(missing_ok=True)
    print(f"ravdess: extracted to {out} ({sum(1 for _ in out.rglob('*.wav'))} wav files)")


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def fetch_kaggle(name: str) -> None:
    out = RAW / name
    if _has_wavs(out):
        print(f"{name}: already present, skipping")
        return
    slug = KAGGLE_SLUGS[name]
    if not _kaggle_available():
        print(
            f"{name}: kaggle CLI not found. Install + configure it, then run:\n"
            f"    pip install kaggle\n"
            f"    # place kaggle.json in ~/.kaggle/ (chmod 600)\n"
            f"    kaggle datasets download -d {slug} -p {out} --unzip\n"
            f"  (Or add the Kaggle dataset '{slug}' to a Kaggle notebook and symlink it.)"
        )
        return
    print(f"{name}: downloading via Kaggle ({slug})...")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(out), "--unzip"],
        check=True,
    )
    print(f"{name}: ready at {out} ({sum(1 for _ in out.rglob('*.wav'))} wav files)")


FETCHERS = {
    "ravdess": fetch_ravdess,
    "tess": lambda: fetch_kaggle("tess"),
    "crema_d": lambda: fetch_kaggle("crema_d"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch corpora for RC1 validation")
    ap.add_argument("datasets", nargs="*", choices=list(FETCHERS), help="datasets to fetch")
    ap.add_argument("--all", action="store_true", help="fetch ravdess + tess + crema_d")
    args = ap.parse_args()

    targets = list(FETCHERS) if args.all else args.datasets
    if not targets:
        ap.error("specify dataset(s) or --all")
    for name in targets:
        FETCHERS[name]()
    print("\nDone. Verify with: python scripts/build_splits.py --dataset ravdess --folds 5")


if __name__ == "__main__":
    main()
