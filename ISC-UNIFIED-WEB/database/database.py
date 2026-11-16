import sqlite3
import os
from pathlib import Path
from werkzeug.security import generate_password_hash
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "database.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'journalist', 'user')),
            full_name TEXT NOT NULL,
            organization TEXT,
            is_approved INTEGER DEFAULT 1, -- Admin and General Users auto-approved; Journalists require admin approval
            failed_logins INTEGER DEFAULT 0,
            locked_until TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 2. Journalist Public Keys Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journalist_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            public_key_pem TEXT NOT NULL,
            key_fingerprint TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'revoked')),
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 3. Media Records Table (Uploaded & Signed by Journalists)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS media_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uploader_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_type TEXT NOT NULL CHECK(file_type IN ('image', 'video')),
            file_size_bytes INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            signature_b64 TEXT NOT NULL,
            title TEXT,
            description TEXT,
            perceptual_hash TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (uploader_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Ensure perceptual_hash column exists if table was created earlier
    cursor.execute("PRAGMA table_info(media_records)")
    m_cols = [c[1] for c in cursor.fetchall()]
    if "perceptual_hash" not in m_cols:
        cursor.execute("ALTER TABLE media_records ADD COLUMN perceptual_hash TEXT")

    # 4. Verification History Table (User Inquiries)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verifier_id INTEGER,
            media_type TEXT NOT NULL CHECK(media_type IN ('image', 'video')),
            filename TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            crypto_status TEXT NOT NULL, -- 'VERIFIED_GENUINE', 'TAMPERED_HASH_MISMATCH', 'INVALID_SIGNATURE', 'UNSIGNED_NO_SIGNATURE'
            signer_name TEXT,
            signer_org TEXT,
            ai_verdict TEXT NOT NULL,
            ai_probability REAL NOT NULL,
            ai_content_pct REAL NOT NULL,
            forensic_details_json TEXT,
            final_assessment TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (verifier_id) REFERENCES users(id) ON DELETE SET NULL
        )
    """)

    # 5. Security Audit Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL, -- 'LOGIN_SUCCESS', 'LOGIN_FAILURE', 'UNAUTHORIZED_ACCESS', 'TAMPER_DETECTED', 'MEDIA_UPLOAD', 'KEY_GEN'
            severity TEXT NOT NULL CHECK(severity IN ('INFO', 'WARNING', 'CRITICAL')),
            user_id INTEGER,
            username TEXT,
            ip_address TEXT,
            description TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # 6. Threat & Anomaly Alerts Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threat_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL, -- 'BRUTE_FORCE_LOGIN', 'SIGNATURE_FORGERY', 'ACCESS_VIOLATION', 'MALICIOUS_UPLOAD'
            threat_level TEXT NOT NULL CHECK(threat_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
            details TEXT NOT NULL,
            ip_address TEXT,
            resolved INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # Seed Default Administrator if not exists
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_pass = generate_password_hash("Admin@12345")
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, full_name, organization, is_approved, created_at)
            VALUES (?, ?, 'admin', 'System Administrator', 'News Integrity Verification Network', 1, ?)
        """, ('admin', admin_pass, datetime.now().isoformat()))

    # Seed Demo Trusted Journalist if not exists
    cursor.execute("SELECT id FROM users WHERE username = 'reuters_editor'")
    if not cursor.fetchone():
        journo_pass = generate_password_hash("Journo@12345")
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, full_name, organization, is_approved, created_at)
            VALUES (?, ?, 'journalist', 'Sarah Jenkins', 'Global News Network / Reuters Wire', 1, ?)
        """, ('reuters_editor', journo_pass, datetime.now().isoformat()))

    conn.commit()
    conn.close()
    print("[Database] SQLite database initialized and seeded successfully at:", DB_PATH)

if __name__ == "__main__":
    init_db()
