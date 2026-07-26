"""Download real corpora into data/raw/ (RAVDESS, TESS, CREMA-D).

Kept as documented pointers rather than automated scraping, because licences differ and
some corpora require manual acceptance. Run the synthetic generator for a zero-download
end-to-end run instead.
"""

from __future__ import annotations

import argparse

SOURCES = {
    "ravdess": "https://zenodo.org/record/1188976  (Audio_Speech_Actors_01-24.zip; speech only)",
    "tess": "https://tspace.library.utoronto.ca/handle/1807/24487",
    "crema_d": "https://github.com/CheyneyComputerScience/CREMA-D  (AudioWAV/)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synthetic", action="store_true", help="Generate the synthetic fixture instead"
    )
    args = ap.parse_args()

    if args.synthetic:
        from pathlib import Path

        from emotionsense.datasets.loaders.synthetic import generate

        path = generate(Path("data/raw/synthetic"))
        print(f"Synthetic corpus generated at {path}")
        return

    print("Manual download required (licences differ). Extract each under data/raw/<name>/:")
    for name, url in SOURCES.items():
        print(f"  - {name:10s} {url}")


if __name__ == "__main__":
    main()
