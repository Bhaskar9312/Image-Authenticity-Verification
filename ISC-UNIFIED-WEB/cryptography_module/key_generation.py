import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def generate_rsa_keypair(key_size=2048):
    """
    Generates a secure RSA-2048 key pair.
    Returns:
        private_key_pem (str): Private key encoded in PEM format (PKCS#8)
        public_key_pem (str): Public key encoded in PEM format (SubjectPublicKeyInfo)
        fingerprint (str): SHA-256 fingerprint of the public key
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode("utf-8")

    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    fingerprint = compute_key_fingerprint(public_pem)
    return private_pem, public_pem, fingerprint

def compute_key_fingerprint(public_key_pem):
    """
    Computes a human-readable SHA-256 fingerprint for a public key PEM string.
    Example format: SHA256:7f83b165...
    """
    clean_key = public_key_pem.strip().encode("utf-8")
    fp = hashlib.sha256(clean_key).hexdigest()
    # Format with colon groupings for forensic display
    formatted = ":".join(fp[i:i+4] for i in range(0, len(fp), 4))
    return f"SHA256:{formatted}"
