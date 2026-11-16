import logging
from datetime import datetime
from pathlib import Path
from database.database import get_db

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_FILE = LOGS_DIR / "security_audit.log"

# Configure standard file logging
file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
logger = logging.getLogger("SECURITY_AUDIT")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(file_handler)

def log_security_event(event_type, severity, description, user_id=None, username=None, ip_address=None):
    """
    Logs security events to both SQLite database and append-only audit log file.
    Guarantees tamper-evident traceability.
    """
    now_str = datetime.now().isoformat()
    # 1. Log to file
    msg = f"event={event_type} user={username or user_id or 'ANON'} ip={ip_address or 'LOCAL'} desc={description}"
    if severity == "CRITICAL":
        logger.critical(msg)
    elif severity == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)

    # 2. Insert into SQLite table
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO security_logs (event_type, severity, user_id, username, ip_address, description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_type, severity, user_id, username, ip_address, description, now_str))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record security log to database: {e}")
