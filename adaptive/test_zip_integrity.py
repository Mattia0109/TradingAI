from __future__ import annotations

import zipfile

import pytest

from adaptive.run_zip_integrity_check import main
from adaptive.zip_integrity import inspect_zip_archive


def test_zip_integrity_reads_every_member_without_extracting(tmp_path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("one.csv", "a,b\n1,2\n")
        archive.writestr("two.csv", "x,y\n3,4\n")

    result = inspect_zip_archive(archive_path)

    assert result.members == 2
    assert result.uncompressed_bytes == 16
    assert not (tmp_path / "one.csv").exists()
    assert main(["--file", str(archive_path)]) == 0


def test_zip_integrity_rejects_truncated_archive(tmp_path, capsys) -> None:
    archive_path = tmp_path / "truncated.zip"
    archive_path.write_bytes(b"PK\x03\x04incomplete")

    with pytest.raises(ValueError, match="Archivio ZIP non valido"):
        inspect_zip_archive(archive_path)

    assert main(["--file", str(archive_path)]) == 2
    assert "ZIP INVALID" in capsys.readouterr().err
