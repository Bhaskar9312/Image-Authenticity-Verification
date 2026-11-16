import hashlib
from pathlib import Path

def calculate_sha256_file(file_path, chunk_size=1048576):
    """
    Computes cryptographic SHA-256 hash of a file using streaming 1MB chunks.
    Guarantees low memory footprint even for gigabyte-scale videos.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()

def calculate_sha256_bytes(data_bytes):
    """Computes SHA-256 hash of in-memory bytes."""
    return hashlib.sha256(data_bytes).hexdigest().lower()
