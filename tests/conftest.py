import urllib.request
from pathlib import Path
import pytest
from backend.core.config import BASE_DIR

FIXTURE_DIR: Path = BASE_DIR / "tests/fixtures"
INGEST_DIR: Path = FIXTURE_DIR / "ingest"
LARGE_FILES_DIR: Path = INGEST_DIR / "large"

# Downloads PDFs and Word docs from internet. Is not commited to repo since its too large.
REMOTE_FIXTURES: dict[str, str] = {
    "test.pdf": "https://arxiv.org/pdf/2408.09869",
    "test.docx": "https://sample-files.com/downloads/documents/docx/sample-files.com-image-document.docx",
}


def _download_if_missing(filename: str, /) -> Path:
    LARGE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    target_path = LARGE_FILES_DIR / filename

    if not target_path.exists():
        url = REMOTE_FIXTURES[filename]
        try:
            _ = urllib.request.urlretrieve(url, filename=target_path)
        except Exception as err:
            pytest.fail(reason=f"Failed downloading {filename} from {url}: {err}")

    return target_path


@pytest.fixture(scope="session")
def fixture_pdf() -> Path:
    return _download_if_missing("test.pdf")


@pytest.fixture(scope="session")
def fixture_docx() -> Path:
    return _download_if_missing("test.docx")


@pytest.fixture(scope="session")
def fixture_md() -> Path:
    p = INGEST_DIR / "test.md"
    if not p.exists():
        pytest.fail(reason=f"Missing required local fixture file: {p}")
    return p


@pytest.fixture(scope="session")
def fixture_xlsx() -> Path:
    p = INGEST_DIR / "test.xlsx"
    if not p.exists():
        pytest.fail(reason=f"Missing required local fixture file: {p}")
    return p
