"""Step 1: download the HMDB51 archive (about 2 GB).

Usage:
    python src/download_data.py
"""

import sys
import urllib.request
from pathlib import Path

from utils import ARCHIVE_PATH

URLS = (
    "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
    "https://web.archive.org/web/20211105033902id_/https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
)
RAR_SIGNATURE = b"Rar!\x1a\x07"   # first bytes of every valid RAR file
BLOCK_SIZE = 1024 * 1024          # read 1 MB at a time


def is_rar_file(path: Path) -> bool:
    """Check the file signature to make sure the file is really a RAR archive."""
    with path.open("rb") as file:
        return file.read(len(RAR_SIGNATURE)) == RAR_SIGNATURE


def show_progress(downloaded: int, total_size: int):
    """Print download progress in the terminal."""
    mb = downloaded / (1024 * 1024)
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        total_mb = total_size / (1024 * 1024)
        status = f"{percent:5.1f}%  {mb:7.1f} / {total_mb:.1f} MB"
    else:
        status = f"{mb:7.1f} MB downloaded"
    sys.stdout.write(f"\r{status}")
    sys.stdout.flush()


def download_archive(url: str, destination: Path):
    """Download one URL into a temporary .part file, check it, then rename it."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    partial_path = destination.with_name(f"{destination.name}.part")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError(f"server returned HTML instead of the archive ({response.geturl()})")

            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else 0
            downloaded = 0

            with partial_path.open("wb") as output:
                while block := response.read(BLOCK_SIZE):
                    output.write(block)
                    downloaded += len(block)
                    show_progress(downloaded, total_size)

        if total_size and downloaded != total_size:
            raise ValueError(f"incomplete download: received {downloaded} of {total_size} bytes")
        if not is_rar_file(partial_path):
            raise ValueError("downloaded file is not a RAR archive")

        partial_path.replace(destination)
    finally:
        partial_path.unlink(missing_ok=True)   # clean up if anything failed


def main():
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Skip the download if a valid archive is already there.
    if ARCHIVE_PATH.exists():
        if is_rar_file(ARCHIVE_PATH):
            print(f"Archive already exists: {ARCHIVE_PATH}")
            return
        print(f"Existing file is invalid and will be replaced: {ARCHIVE_PATH}")

    # Try each URL in order until one works.
    errors = []
    for url in URLS:
        print(f"Downloading HMDB51 from {url}...")
        try:
            download_archive(url, ARCHIVE_PATH)
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {error}")
            print(f"\nDownload failed: {error}")
            continue

        print(f"\nDone! Saved valid archive to: {ARCHIVE_PATH}")
        return

    raise RuntimeError("Unable to download a valid HMDB51 archive:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
