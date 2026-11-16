import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from .security_logging import log_security_event
from .threat_detection import record_threat_alert

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS

# Max sizes
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Magic Bytes Signatures
MAGIC_SIGNATURES = {
    # JPEG
    b"\xFF\xD8\xFF": "image",
    # PNG
    b"\x89PNG\r\n\x1a\n": "image",
    # WEBP (RIFF....WEBP)
    b"RIFF": "image",
    # MP4 / MOV (typically has 'ftyp' in first 16 bytes)
    b"ftyp": "video",
    # Matroska / MKV / WebM
    b"\x1A\x45\xDF\xA3": "video",
    # AVI
    b"AVI ": "video"
}

def check_magic_bytes(file_stream):
    """
    Reads the first 32 bytes of the file stream to verify actual binary type against spoofed extensions.
    """
    file_stream.seek(0)
    header = file_stream.read(32)
    file_stream.seek(0)

    # Check known signatures
    for sig, mtype in MAGIC_SIGNATURES.items():
        if sig in header:
            return True, mtype

    # Fallback check for ftyp anywhere in first 32 bytes
    if b"ftyp" in header or b"moov" in header:
        return True, "video"

    return False, "unknown"

def validate_uploaded_file(file_storage, ip_address=None):
    """
    Validates uploaded file against:
    1. Null byte / path traversal injection
    2. Allowed extensions
    3. File size constraints
    4. Magic binary header check
    Returns: (is_valid: bool, error_msg: str, media_type: str)
    """
    if not file_storage or not file_storage.filename:
        return False, "No file selected.", None

    orig_name = file_storage.filename
    if "\x00" in orig_name or ".." in orig_name or "/" in orig_name or "\\" in orig_name:
        record_threat_alert(
            alert_type="MALICIOUS_UPLOAD",
            threat_level="CRITICAL",
            details=f"Path traversal / null-byte attack detected in filename: '{orig_name}'",
            ip_address=ip_address
        )
        return False, "Malicious filename detected.", None

    ext = Path(orig_name).suffix.lower()
    if ext not in ALL_ALLOWED_EXTS:
        return False, f"File format '{ext}' is not supported. Supported: JPG, PNG, WEBP, MP4, MOV, MKV, AVI.", None

    expected_type = "image" if ext in ALLOWED_IMAGE_EXTS else "video"

    # Check file size by seeking
    file_storage.seek(0, os.SEEK_END)
    size_bytes = file_storage.tell()
    file_storage.seek(0)

    max_size = MAX_IMAGE_SIZE_BYTES if expected_type == "image" else MAX_VIDEO_SIZE_BYTES
    if size_bytes > max_size:
        max_mb = max_size // (1024 * 1024)
        return False, f"File exceeds maximum allowed size ({max_mb} MB).", None

    if size_bytes == 0:
        return False, "Uploaded file is empty (0 bytes).", None

    # Magic Bytes Validation
    is_valid_magic, detected_type = check_magic_bytes(file_storage.stream)
    if not is_valid_magic:
        record_threat_alert(
            alert_type="MALICIOUS_UPLOAD",
            threat_level="HIGH",
            details=f"File extension spoofing detected: '{orig_name}' with invalid magic bytes header.",
            ip_address=ip_address
        )
        return False, "Security error: File binary signature does not match valid image or video formats.", None

    return True, None, expected_type

def sanitize_filename(original_filename):
    """
    Generates a secure, random server-side filename while preserving sanitized extension.
    Prevents file overwriting and direct script execution.
    """
    clean_base = secure_filename(original_filename)
    ext = Path(clean_base).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return unique_name
