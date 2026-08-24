"""CLI per il controllo locale di un singolo archivio ZIP."""

from __future__ import annotations

import argparse
import sys

from adaptive.zip_integrity import inspect_zip_archive


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Verifica struttura e CRC di uno ZIP senza estrarlo."
    )
    parser.add_argument("--file", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    try:
        result = inspect_zip_archive(arguments.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ZIP INVALID: {exc}", file=sys.stderr)
        return 2
    print(
        "ZIP OK: "
        f"{result.path} | membri={result.members} | "
        f"byte_non_compressi={result.uncompressed_bytes}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
