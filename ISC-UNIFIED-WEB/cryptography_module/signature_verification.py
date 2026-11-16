import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
from .hashing import calculate_sha256_file

def verify_media_signature(sha256_hex, signature_b64, public_key_pem):
    """
    Verifies an RSA-PSS digital signature against the media SHA-256 hash.
    Returns:
        is_valid (bool): True if signature is cryptographically valid, False otherwise.
    """
    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode("utf-8"),
            backend=default_backend()
        )

        signature_bytes = base64.b64decode(signature_b64)
        data_to_verify = sha256_hex.strip().lower().encode("utf-8")

        public_key.verify(
            signature_bytes,
            data_to_verify,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except (InvalidSignature, ValueError, Exception):
        return False

def verify_hash_integrity(file_path, expected_sha256_hex):
    """
    Checks if a local media file's current SHA-256 matches an expected reference hash.
    Detects unauthorized file modifications / bit flipping.
    """
    current_hash = calculate_sha256_file(file_path)
    return current_hash.lower() == expected_sha256_hex.strip().lower(), current_hash
