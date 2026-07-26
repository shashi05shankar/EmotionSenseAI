"""Seed the serving environment: train + register a default model, and (optionally)
generate the admin password hash to configure JWT auth.

For the Lean Core runnable path this seeds the filesystem registry (no DB needed). In the
DB-backed production path this also inserts the admin user and dataset rows.

Usage:
    python scripts/seed_db.py
    python scripts/seed_db.py --admin-password 'your-strong-password'
        -> prints the bcrypt hash to set as ESA_ADMIN_PASSWORD_HASH (no credential is
           stored in code; you place the hash in your environment).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from emotionsense.backend.security import hash_password


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--admin-password",
        help="Print the bcrypt hash for this password (set it as ESA_ADMIN_PASSWORD_HASH).",
    )
    ap.add_argument("--skip-model", action="store_true", help="Only handle the admin hash.")
    args = ap.parse_args()

    if args.admin_password:
        digest = hash_password(args.admin_password)
        print("Set this in your environment (do NOT commit it):")
        print(f'  ESA_ADMIN_PASSWORD_HASH="{digest}"')

    if args.skip_model:
        return

    print("Seeding: training + registering svm-mfcc on synthetic as production default...")
    subprocess.run(
        [
            sys.executable,
            "scripts/train_and_register.py",
            "--model",
            "svm",
            "--dataset",
            "synthetic",
            "--promote",
        ],
        check=True,
    )
    print("Done. Default model registered. Start the API: uvicorn emotionsense.backend.main:app")


if __name__ == "__main__":
    main()
