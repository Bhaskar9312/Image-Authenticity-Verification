import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from .hashing import calculate_sha256_file

def sign_media_hash(sha256_hex, private_key_pem):
    """
    Signs a media SHA-256 hash string using RSA-PSS padding and MGF1(SHA-256).
    Returns Base64-encoded digital signature string.
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"),
        password=None,
        backend=default_backend()
    )

    data_to_sign = sha256_hex.strip().lower().encode("utf-8")

    signature = private_key.sign(
        data_to_sign,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return base64.b64encode(signature).decode("utf-8")

def sign_media_file(file_path, private_key_pem):
    """
    Calculates SHA-256 hash of a media file and produces an RSA-PSS digital signature.
    Returns: (sha256_hex, signature_b64)
    """
    sha256_hex = calculate_sha256_file(file_path)
    sig_b64 = sign_media_hash(sha256_hex, private_key_pem)
    return sha256_hex, sig_b64
