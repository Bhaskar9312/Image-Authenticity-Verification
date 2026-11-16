from datetime import datetime
from database.database import get_db
from .security_logging import log_security_event

def record_threat_alert(alert_type, threat_level, details, ip_address=None):
    """
    Records an active security threat / anomaly alert in SQLite and triggers logging.
    """
    now_str = datetime.now().isoformat()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO threat_alerts (alert_type, threat_level, details, ip_address, resolved, created_at)
        VALUES (?, ?, ?, ?, 0, ?)
    """, (alert_type, threat_level, details, ip_address, now_str))
    conn.commit()
    conn.close()

    log_security_event(
        event_type=f"ALERT_{alert_type}",
        severity="CRITICAL" if threat_level in ("HIGH", "CRITICAL") else "WARNING",
        description=f"THREAT DETECTED: {details}",
        ip_address=ip_address
    )

def detect_login_threat(username, current_failed_count, ip_address=None):
    """
    Detects brute force attacks if failed login count exceeds threshold.
    """
    if current_failed_count >= 5:
        record_threat_alert(
            alert_type="BRUTE_FORCE_LOGIN",
            threat_level="HIGH",
            details=f"Account '{username}' locked out after {current_failed_count} repeated failed login attempts.",
            ip_address=ip_address
        )
