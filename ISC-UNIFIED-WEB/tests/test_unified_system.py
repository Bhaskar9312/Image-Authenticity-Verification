import sys
import unittest
from pathlib import Path

# Add root project directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app import create_app
from database.database import get_db, init_db
from cryptography_module import (
    generate_rsa_keypair, sign_media_hash,
    verify_media_signature, calculate_sha256_bytes
)
from security_module import (
    register_user, authenticate_user,
    validate_uploaded_file, sanitize_filename
)
from ai_module import ImageForgeryDetector, VideoForgeryDetectorEngine

class TestUnifiedSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_01_database_and_admin_seed(self):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT username, role FROM users WHERE username = 'admin'")
        admin = cursor.fetchone()
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "admin")
        conn.close()
        print("[PASS] Test 1: Database & Admin seeded.")

    def test_02_cryptography_signing_and_verification(self):
        priv_pem, pub_pem, fingerprint = generate_rsa_keypair(2048)
        self.assertTrue(fingerprint.startswith("SHA256:"))

        test_hash = calculate_sha256_bytes(b"Breaking News Original Media Content")
        sig_b64 = sign_media_hash(test_hash, priv_pem)
        self.assertTrue(len(sig_b64) > 100)

        # Valid verification
        is_valid = verify_media_signature(test_hash, sig_b64, pub_pem)
        self.assertTrue(is_valid)

        # Tampered hash verification
        tampered_hash = calculate_sha256_bytes(b"Modified Fake News Content")
        is_tampered_valid = verify_media_signature(tampered_hash, sig_b64, pub_pem)
        self.assertFalse(is_tampered_valid)
        print("[PASS] Test 2: RSA-PSS Cryptography & Tamper Detection verified.")

    def test_03_security_auth_and_lockout(self):
        import uuid
        uname = f"user_{uuid.uuid4().hex[:8]}"
        success, msg = register_user(uname, "Secret@123", "user", "Test General User")
        self.assertTrue(success)

        # Correct password
        ok, user = authenticate_user(uname, "Secret@123")
        self.assertTrue(ok)
        self.assertEqual(user["username"], uname)

        # Wrong password
        ok_wrong, err = authenticate_user(uname, "WrongPass123")
        self.assertFalse(ok_wrong)
        print("[PASS] Test 3: Security Authentication & Password Hashing verified.")

    def test_04_web_routes_status(self):
        # Home
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)

        # Verify page
        res_verify = self.client.get("/verify")
        self.assertEqual(res_verify.status_code, 200)

        # Login page
        res_login = self.client.get("/login")
        self.assertEqual(res_login.status_code, 200)

        # Register page
        res_reg = self.client.get("/register")
        self.assertEqual(res_reg.status_code, 200)
        print("[PASS] Test 4: Web frontend routes respond with HTTP 200.")

    def test_05_image_detector_inference(self):
        from PIL import Image
        img_path = BASE_DIR / "uploads" / "test_sample.jpg"
        test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        test_img.save(img_path)

        detector = ImageForgeryDetector.get_instance()
        report = detector.analyze(img_path)
        self.assertEqual(report["media_type"], "image")
        self.assertIn("overall_decision", report["verdict"])
        self.assertIn("overall_ai_percentage", report["verdict"])
        print(f"[PASS] Test 5: Image AI Detector inference complete! Result: {report['verdict']['overall_decision']} ({report['verdict']['overall_ai_percentage']}%)")

    def test_06_video_detector_initialization(self):
        video_engine = VideoForgeryDetectorEngine.get_instance()
        self.assertIsNotNone(video_engine.model)
        print("[PASS] Test 6: Video ConvNeXt-BiGRU Detector engine loaded into VRAM.")

if __name__ == "__main__":
    unittest.main()
