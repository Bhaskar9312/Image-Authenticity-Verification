from functools import wraps
from flask import session, redirect, url_for, flash, request, abort
from .security_logging import log_security_event
from .threat_detection import record_threat_alert

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("routes.login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("routes.login"))
            
            user_role = session.get("role")
            if user_role not in allowed_roles:
                log_security_event(
                    event_type="UNAUTHORIZED_ACCESS_ATTEMPT",
                    severity="WARNING",
                    description=f"User '{session.get('username')}' (Role: {user_role}) attempted to access restricted endpoint: {request.path}",
                    user_id=session.get("user_id"),
                    username=session.get("username"),
                    ip_address=request.remote_addr
                )
                record_threat_alert(
                    alert_type="ACCESS_VIOLATION",
                    threat_level="MEDIUM",
                    details=f"Privilege violation: User '{session.get('username')}' attempted to access '{request.path}' requiring roles: {allowed_roles}",
                    ip_address=request.remote_addr
                )
                flash("Access denied: You do not have permission to view this resource.", "danger")
                return redirect(url_for("routes.index"))

            # If role is journalist, ensure account is approved
            if user_role == "journalist" and not session.get("is_approved", 0):
                flash("Your journalist account is pending administrator approval.", "warning")
                return redirect(url_for("routes.index"))

            return f(*args, **kwargs)
        return decorated_function
    return decorator

def admin_required(f):
    return role_required("admin")(f)

def journalist_required(f):
    return role_required("journalist", "admin")(f)
