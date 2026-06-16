from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from backend.core.ingest import verify_file_integrity


def _write_file(path: Path, data: bytes) -> Path:
    _ = path.write_bytes(data)
    return path


class TestVerifyFileIntegrity:
    def test_pdf_valid(self, tmp_path: Path) -> None:
        pdf_path = _write_file(path=tmp_path / "valid.pdf", data=b"%PDF-1.7\ncontent")
        assert verify_file_integrity(path=pdf_path) == pdf_path

    def test_pdf_invalid_signature(self, tmp_path: Path) -> None:
        pdf_path = _write_file(path=tmp_path / "invalid.pdf", data=b"NOTPDF")
        with pytest.raises(ValueError, match="File signature mismatch"):
            _ = verify_file_integrity(path=pdf_path)

    def test_docx_valid(self, tmp_path: Path) -> None:
        docx_path = _write_file(path=tmp_path / "valid.docx", data=b"PK\x03\x04content")
        assert verify_file_integrity(path=docx_path) == docx_path

    def test_docx_invalid_signature(self, tmp_path: Path) -> None:
        docx_path = _write_file(path=tmp_path / "invalid.docx", data=b"NOTPK")
        with pytest.raises(ValueError, match="File signature mismatch"):
            _ = verify_file_integrity(path=docx_path)

    def test_unreadable_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bad_path = tmp_path / "bad.pdf"
        bad_path.touch()

        _ = mocker.patch.object(
            target=Path,
            attribute="open",
            side_effect=OSError("permission denied"),
        )

        with pytest.raises(ValueError, match="File validation failed"):
            _ = verify_file_integrity(path=bad_path)
