from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from database.database import get_db
from .security_logging import log_security_event
from .threat_detection import detect_login_threat

MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15

def register_user(username, password, role, full_name, organization=None, ip_address=None):
    """
    Registers a new user with password hashing and initial approval state.
    Journalist role requires administrator approval.
    """
    username = username.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if role not in ("user", "journalist", "admin"):
        return False, "Invalid role specified."

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username is already registered."

    pass_hash = generate_password_hash(password)
    # Journalists require manual admin approval before signing privileges
    is_approved = 1 if role in ("user", "admin") else 0
    now_str = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO users (username, password_hash, role, full_name, organization, is_approved, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, pass_hash, role, full_name, organization, is_approved, now_str))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()

    log_security_event(
        event_type="USER_REGISTER",
        severity="INFO",
        description=f"New user registered: {username} (Role: {role}, Approved: {bool(is_approved)})",
        user_id=user_id,
        username=username,
        ip_address=ip_address
    )

    msg = "Registration successful! You may now log in." if is_approved else "Journalist account created! Awaiting administrator approval before digital signing is activated."
    return True, msg

def authenticate_user(username, password, ip_address=None):
    """
    Authenticates a user with brute-force lockout protection.
    Returns: (success: bool, user_data_or_error_msg)
    """
    username = username.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        log_security_event(
            event_type="LOGIN_FAILED_UNKNOWN_USER",
            severity="WARNING",
            description=f"Failed login attempt for non-existent username: {username}",
            ip_address=ip_address
        )
        return False, "Invalid username or password."

    # Check Account Lockout
    if user["locked_until"]:
        locked_time = datetime.fromisoformat(user["locked_until"])
        if datetime.now() < locked_time:
            conn.close()
            remaining = int((locked_time - datetime.now()).total_seconds() // 60) + 1
            return False, f"Account temporarily locked due to repeated failed logins. Please try again in {remaining} minute(s)."
        else:
            # Lockout expired, reset
            cursor.execute("UPDATE users SET locked_until = NULL, failed_logins = 0 WHERE id = ?", (user["id"],))
            conn.commit()

    # Verify Password
    if not check_password_hash(user["password_hash"], password):
        failed_count = user["failed_logins"] + 1
        locked_until_str = None
        if failed_count >= MAX_FAILED_LOGINS:
            locked_until_str = (datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            detect_login_threat(username, failed_count, ip_address=ip_address)

        cursor.execute("UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                       (failed_count, locked_until_str, user["id"]))
        conn.commit()
        conn.close()

        log_security_event(
            event_type="LOGIN_FAILED",
            severity="WARNING",
            description=f"Failed login attempt for user: {username} (Failed count: {failed_count})",
            user_id=user["id"],
            username=username,
            ip_address=ip_address
        )

        if locked_until_str:
            return False, f"Too many failed login attempts. Account locked for {LOCKOUT_MINUTES} minutes."
        return False, f"Invalid username or password. ({MAX_FAILED_LOGINS - failed_count} attempts remaining)"

    # Reset failed logins on success
    cursor.execute("UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ?", (user["id"],))
    conn.commit()

    user_dict = dict(user)
    conn.close()

    log_security_event(
        event_type="LOGIN_SUCCESS",
        severity="INFO",
        description=f"User logged in successfully: {username} (Role: {user['role']})",
        user_id=user["id"],
        username=username,
        ip_address=ip_address
    )

    return True, user_dict

def lock_user_account(user_id, minutes=LOCKOUT_MINUTES):
    conn = get_db()
    cursor = conn.cursor()
    locked_str = (datetime.now() + timedelta(minutes=minutes)).isoformat()
    cursor.execute("UPDATE users SET locked_until = ? WHERE id = ?", (locked_str, user_id))
    conn.commit()
    conn.close()

def reset_failed_logins(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
