"""Controllo locale e non distruttivo dell'integrita' di un archivio ZIP."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZipIntegrityResult:
    path: str
    members: int
    uncompressed_bytes: int


def inspect_zip_archive(path: str | Path) -> ZipIntegrityResult:
    """Legge tutti i membri per verificare struttura e CRC, senza estrarli."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Archivio ZIP non trovato: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            bad_member = archive.testzip()
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Archivio ZIP non valido: {source}: {exc}") from exc
    if bad_member is not None:
        raise ValueError(
            f"Archivio ZIP non valido: {source}: membro corrotto={bad_member}"
        )
    if not members:
        raise ValueError(f"Archivio ZIP vuoto: {source}")
    return ZipIntegrityResult(
        path=str(source.resolve()),
        members=len(members),
        uncompressed_bytes=sum(int(member.file_size) for member in members),
    )
