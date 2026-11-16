import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  VERIMEDIA AI: NEWS AUTHENTICITY & FORGERY VERIFICATION SYSTEM")
    print("  Converging AI Forgery Detection, Cryptography & Information Security")
    print("=" * 75)
    print("  * Local Web URL: http://127.0.0.1:5000")
    print("  * Default Admin: admin / Admin@12345")
    print("  * Demo Journalist: reuters_editor / Journo@12345")
    print("=" * 75 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
