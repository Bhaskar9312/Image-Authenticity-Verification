import os
from pathlib import Path
from flask import Flask, session, g
from database.database import init_db, get_db

BASE_DIR = Path(__file__).resolve().parent.parent

def create_app():
    template_dir = BASE_DIR / "frontend" / "templates"
    static_dir = BASE_DIR / "frontend" / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )

    # Secure Session Configuration
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ISC-UNIFIED-NEWS-AUTHENTICITY-KEY-2026-X89")
    app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024 # 120 MB max upload
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # Ensure required runtime folders exist
    for sub in ["uploads", "signatures", "keys", "logs"]:
        (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Initialize SQLite database
    init_db()

    # Context processor to expose logged in user to all templates
    @app.context_processor
    def inject_user():
        user = None
        if "user_id" in session:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, role, full_name, organization, is_approved FROM users WHERE id = ?", (session["user_id"],))
            user = cursor.fetchone()
            conn.close()
        return dict(current_user=user)

    # Register Blueprint routes
    from .routes import routes_bp
    app.register_blueprint(routes_bp)

    return app
