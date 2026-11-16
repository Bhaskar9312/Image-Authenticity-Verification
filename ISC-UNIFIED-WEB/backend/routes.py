import os
import json
from pathlib import Path
from datetime import datetime
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, session, jsonify, send_file, current_app
)

from database.database import get_db
from security_module import (
    register_user, authenticate_user, login_required,
    role_required, admin_required, journalist_required,
    validate_uploaded_file, sanitize_filename, log_security_event,
    record_threat_alert
)
from cryptography_module import (
    calculate_sha256_file, generate_rsa_keypair,
    sign_media_file, verify_media_signature, verify_hash_integrity,
    find_registered_original, quantify_differential_manipulation,
    compute_media_fingerprint
)
from ai_module import ImageForgeryDetector, VideoForgeryDetectorEngine

routes_bp = Blueprint("routes", __name__)
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------
# 1. PUBLIC & AUTHENTICATION ROUTES
# -------------------------------------------------------------

@routes_bp.route("/")
def index():
    """Landing page showcasing the 3-domain authenticity framework."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM verification_history")
    total_verifications = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM media_records")
    total_signed_media = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'journalist' AND is_approved = 1")
    total_trusted_journalists = cursor.fetchone()[0]
    conn.close()

    return render_template(
        "index.html",
        total_verifications=total_verifications,
        total_signed_media=total_signed_media,
        total_trusted_journalists=total_trusted_journalists
    )

@routes_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "user")
        full_name = request.form.get("full_name", "").strip()
        organization = request.form.get("organization", "").strip()
        ip_addr = request.remote_addr

        success, msg = register_user(
            username=username,
            password=password,
            role=role,
            full_name=full_name,
            organization=organization,
            ip_address=ip_addr
        )

        if success:
            flash(msg, "success")
            return redirect(url_for("routes.login"))
        else:
            flash(msg, "danger")

    return render_template("register.html")

@routes_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        ip_addr = request.remote_addr

        success, res = authenticate_user(username, password, ip_address=ip_addr)
        if success:
            session["user_id"] = res["id"]
            session["username"] = res["username"]
            session["role"] = res["role"]
            session["full_name"] = res["full_name"]
            session["is_approved"] = res["is_approved"]

            flash(f"Welcome back, {res['full_name']}!", "success")
            if res["role"] == "admin":
                return redirect(url_for("routes.admin_dashboard"))
            elif res["role"] == "journalist":
                return redirect(url_for("routes.journalist_dashboard"))
            else:
                return redirect(url_for("routes.verify"))
        else:
            flash(res, "danger")

    return render_template("login.html")

@routes_bp.route("/logout")
def logout():
    user_id = session.get("user_id")
    username = session.get("username")
    log_security_event(
        event_type="USER_LOGOUT",
        severity="INFO",
        description=f"User logged out: {username}",
        user_id=user_id,
        username=username,
        ip_address=request.remote_addr
    )
    session.clear()
    flash("You have been securely logged out.", "info")
    return redirect(url_for("routes.index"))

# -------------------------------------------------------------
# 2. MEDIA VERIFICATION PORTAL (PHOTO & VIDEO)
# -------------------------------------------------------------

@routes_bp.route("/verify", methods=["GET", "POST"])
def verify():
    """Unified media verification endpoint supporting both photos and videos."""
    if request.method == "POST":
        if "media_file" not in request.files:
            flash("No media file uploaded.", "danger")
            return redirect(request.url)

        file = request.files["media_file"]
        is_valid, err_msg, media_type = validate_uploaded_file(file, ip_address=request.remote_addr)
        if not is_valid:
            flash(err_msg, "danger")
            return redirect(request.url)

        # Secure Storage
        safe_name = sanitize_filename(file.filename)
        save_path = Path(current_app.config["UPLOAD_FOLDER"]) / safe_name
        file.save(str(save_path))

        # A. Calculate Media SHA-256
        sha256_hash = calculate_sha256_file(save_path)

        # B. Optional Signature & Public Key Input
        sig_file = request.files.get("signature_file")
        sig_text = request.form.get("signature_text", "").strip()
        pub_key_file = request.files.get("public_key_file")
        pub_key_text = request.form.get("public_key_text", "").strip()

        signature_b64 = None
        if sig_file and sig_file.filename:
            signature_b64 = sig_file.read().decode("utf-8", errors="ignore").strip()
        elif sig_text:
            signature_b64 = sig_text

        public_key_pem = None
        if pub_key_file and pub_key_file.filename:
            public_key_pem = pub_key_file.read().decode("utf-8", errors="ignore").strip()
        elif pub_key_text:
            public_key_pem = pub_key_text

        # C. Cryptographic Source & Integrity Check
        conn = get_db()
        cursor = conn.cursor()

        # Check if this media hash is already registered by a known journalist in the database
        cursor.execute("""
            SELECT m.*, u.full_name as signer_name, u.organization as signer_org, k.public_key_pem as reg_pubkey
            FROM media_records m
            JOIN users u ON m.uploader_id = u.id
            LEFT JOIN journalist_keys k ON k.user_id = u.id AND k.status = 'active'
            WHERE m.sha256_hash = ?
        """, (sha256_hash,))
        matched_record = cursor.fetchone()

        crypto_status = "UNSIGNED_NO_SIGNATURE"
        signer_name = None
        signer_org = None
        diff_report = None
        matched_original_record = None

        if matched_record:
            # File matches a registered journalist original upload!
            registered_sig = matched_record["signature_b64"]
            registered_pubkey = matched_record["reg_pubkey"]
            if registered_pubkey and verify_media_signature(sha256_hash, registered_sig, registered_pubkey):
                crypto_status = "VERIFIED_GENUINE"
                signer_name = matched_record["signer_name"]
                signer_org = matched_record["signer_org"]
        elif signature_b64 and public_key_pem:
            # User provided a custom signature & public key
            if verify_media_signature(sha256_hash, signature_b64, public_key_pem):
                crypto_status = "VERIFIED_GENUINE"
                signer_name = "Independent Signer (Public Key Verified)"
                signer_org = "External Entity"
            else:
                crypto_status = "TAMPERED_OR_INVALID_SIGNATURE"
                record_threat_alert(
                    alert_type="SIGNATURE_FORGERY",
                    threat_level="HIGH",
                    details=f"Digital signature verification failed for file: {file.filename} (Hash: {sha256_hash[:16]}...)",
                    ip_address=request.remote_addr
                )
        else:
            # Perceptual Provenance Matching:
            # Check if this media is a modified derivative of ANY registered journalist original in our database!
            p_record, sim_pct = find_registered_original(save_path, media_type, conn, min_similarity=75.0)
            if p_record:
                crypto_status = "TAMPERED_HASH_MISMATCH"
                matched_original_record = p_record
                signer_name = p_record["signer_name"]
                signer_org = p_record["signer_org"]

                diff_cache_dir = Path(current_app.static_folder) / "diff_cache"
                diff_report = quantify_differential_manipulation(
                    query_path=save_path,
                    orig_path=p_record["file_path"],
                    media_type=media_type,
                    cache_dir=str(diff_cache_dir)
                )

                record_threat_alert(
                    alert_type="MEDIA_TAMPERED",
                    threat_level="HIGH",
                    details=(
                        f"Media '{file.filename}' is an altered derivative of registered news asset '{p_record['original_filename']}' "
                        f"by {signer_name} ({signer_org}). Cryptographic SHA-256 altered! Altered frames: {diff_report['manipulated_frames_pct']}%."
                    ),
                    ip_address=request.remote_addr
                )
            elif signature_b64 and not public_key_pem:
                crypto_status = "UNVERIFIED_MISSING_PUBLIC_KEY"
            else:
                crypto_status = "UNSIGNED_NO_SIGNATURE"

        # D. Run AI Forgery & Manipulation Detection (Using Trained Domain Models)
        ai_report = None
        if media_type == "image":
            image_detector = ImageForgeryDetector.get_instance()
            ai_report = image_detector.analyze(save_path)
            model_name = "CLIP ViT-L/14 Calibrated Ensemble (Image Model)"
            ai_verdict = ai_report["verdict"]["overall_decision"]
            ai_prob = float(ai_report["verdict"]["overall_ai_percentage"])
            real_prob = float(ai_report["verdict"].get("overall_real_percentage", round(100.0 - ai_prob, 1)))
            ai_content_pct = float(ai_report["content_composition"]["ai_content_percentage"])
        else:
            video_detector = VideoForgeryDetectorEngine.get_instance()
            ai_report = video_detector.analyze(save_path, sample_fps=2.0)
            model_name = "ConvNeXt-Tiny + 8x8 DCT Frequency + BiGRU Temporal (Video Model)"
            ai_verdict = ai_report["verdict"]["overall_decision"]
            ai_prob = float(ai_report["verdict"]["overall_ai_percentage"])
            real_prob = float(ai_report["verdict"].get("overall_real_percentage", round(100.0 - ai_prob, 1)))
            ai_content_pct = float(ai_report["content_composition"]["ai_content_percentage"])

        # Retain pure AI Model metrics (never overwritten by tampering differential)
        ai_report["ai_model_metrics"] = {
            "model_name": model_name,
            "media_type": media_type,
            "ai_percentage": ai_prob,
            "real_percentage": real_prob,
            "verdict": ai_verdict,
            "confidence": ai_report["verdict"].get("confidence_level", "Standard")
        }

        if diff_report:
            ai_report["differential_analysis"] = diff_report
            ai_report["matched_original_title"] = matched_original_record["original_filename"]
            ai_report["matched_original_hash"] = matched_original_record["sha256_hash"]
            ai_report["matched_similarity_pct"] = sim_pct

            if diff_report.get("tampered_intervals"):
                ai_report["suspicious_segments"] = diff_report["tampered_intervals"] + ai_report.get("suspicious_segments", [])

        # E. Synthesize Combined Authenticity Assessment
        # IMPORTANT: Cryptography and AI provide separate, complementary forensic layers!
        if crypto_status == "VERIFIED_GENUINE":
            if "REAL" in ai_verdict or "AUTHENTIC" in ai_verdict:
                final_assessment = (
                    f"AUTHENTIC & VERIFIED: Cryptographic digital signature is valid from trusted source ({signer_name}, {signer_org}) with 0 bit modification. "
                    f"AI {media_type.upper()} model confirms genuine capture ({real_prob}% Real, {ai_prob}% AI)."
                )
            else:
                final_assessment = (
                    f"TAMPERED / MISLEADING: Valid cryptographic signature matches source ({signer_name}), "
                    f"but AI forensic model flags synthetic manipulation ({ai_prob}% AI probability)."
                )
        elif crypto_status == "TAMPERED_HASH_MISMATCH" and matched_original_record:
            if media_type == "image":
                final_assessment = (
                    f"CRITICAL TAMPER WARNING: Cryptographic hash mismatch! Photograph was modified after signing by {signer_name} ({signer_org}). "
                    f"Visual differential analysis detected {diff_report['avg_pixel_manipulation_pct']}% of pixels altered ({diff_report.get('tamper_classification', 'Background Replacement / Splicing')}). "
                    f"AI {media_type.upper()} model evaluation: {ai_verdict} ({ai_prob}% AI, {real_prob}% Real)."
                )
            else:
                final_assessment = (
                    f"CRITICAL TAMPER WARNING: Cryptographic hash mismatch! Video was modified after signing by {signer_name} ({signer_org}). "
                    f"Visual tampering analysis shows {diff_report['manipulated_frames_pct']}% of frames altered with foreign overlays / stickers. "
                    f"AI {media_type.upper()} model evaluation of base media: {ai_verdict} ({ai_prob}% AI, {real_prob}% Real)."
                )
        elif crypto_status == "TAMPERED_OR_INVALID_SIGNATURE":
            final_assessment = (
                f"CRYPTOGRAPHIC INTEGRITY FAILED: Media does not match provided digital signature (file modified since signing). "
                f"AI {media_type.upper()} model score: {ai_prob}% AI, {real_prob}% Real ({ai_verdict})."
            )
        else:
            if "REAL" in ai_verdict or "AUTHENTIC" in ai_verdict:
                final_assessment = (
                    f"AI AUTHENTIC (UNSIGNED): AI {media_type.upper()} model confirms authentic camera capture ({real_prob}% Real, {ai_prob}% AI). "
                    f"No cryptographic digital signature was provided for source provenance."
                )
            else:
                final_assessment = (
                    f"AI MANIPULATED (UNSIGNED): High probability of synthetic AI generation / deepfake manipulation ({ai_prob}% AI, {real_prob}% Real). "
                    f"No valid cryptographic signature found."
                )

        # F. Record into Verification History
        verifier_id = session.get("user_id")
        now_str = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO verification_history (
                verifier_id, media_type, filename, sha256_hash, crypto_status,
                signer_name, signer_org, ai_verdict, ai_probability, ai_content_pct,
                forensic_details_json, final_assessment, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            verifier_id, media_type, file.filename, sha256_hash, crypto_status,
            signer_name, signer_org, ai_verdict, ai_prob, ai_content_pct,
            json.dumps(ai_report), final_assessment, now_str
        ))
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()

        log_security_event(
            event_type="MEDIA_VERIFIED",
            severity="INFO",
            description=f"Media verified: {file.filename} ({media_type}) | Crypto: {crypto_status} | AI: {ai_verdict} ({ai_prob}%)",
            user_id=verifier_id,
            username=session.get("username"),
            ip_address=request.remote_addr
        )

        return redirect(url_for("routes.view_report", report_id=report_id))

    return render_template("verify.html")

@routes_bp.route("/report/<int:report_id>")
def view_report(report_id):
    """Displays comprehensive report combining Cryptography, AI, and Information Security."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM verification_history WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        flash("Verification report not found.", "danger")
        return redirect(url_for("routes.verify"))

    report = dict(row)
    report["forensic_details"] = json.loads(report["forensic_details_json"])
    return render_template("report.html", report=report)

# -------------------------------------------------------------
# 3. JOURNALIST WORKFLOW & DIGITAL SIGNATURES
# -------------------------------------------------------------

@routes_bp.route("/journalist/dashboard")
@journalist_required
def journalist_dashboard():
    user_id = session["user_id"]
    conn = get_db()
    cursor = conn.cursor()

    # Active RSA Keys
    cursor.execute("SELECT * FROM journalist_keys WHERE user_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1", (user_id,))
    active_key = cursor.fetchone()

    # Media Records signed by this journalist
    cursor.execute("SELECT * FROM media_records WHERE uploader_id = ? ORDER BY id DESC", (user_id,))
    signed_records = cursor.fetchall()
    conn.close()

    return render_template("journalist_dashboard.html", active_key=active_key, signed_records=signed_records)

@routes_bp.route("/journalist/generate-keys", methods=["POST"])
@journalist_required
def journalist_generate_keys():
    user_id = session["user_id"]
    priv_pem, pub_pem, fingerprint = generate_rsa_keypair(key_size=2048)
    now_str = datetime.now().isoformat()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE journalist_keys SET status = 'revoked' WHERE user_id = ?", (user_id,))
    cursor.execute("""
        INSERT INTO journalist_keys (user_id, public_key_pem, key_fingerprint, status, created_at)
        VALUES (?, ?, ?, 'active', ?)
    """, (user_id, pub_pem, fingerprint, now_str))
    conn.commit()
    conn.close()

    log_security_event(
        event_type="KEYPAIR_GENERATED",
        severity="INFO",
        description=f"Journalist '{session.get('username')}' generated new RSA-2048 keypair (Fingerprint: {fingerprint})",
        user_id=user_id,
        username=session.get("username"),
        ip_address=request.remote_addr
    )

    # Save private key to session for instant signing and download
    session["download_private_key"] = priv_pem
    session["active_private_key"] = priv_pem
    flash("New RSA-2048 keypair generated successfully! Download your private key or sign media directly.", "success")
    return redirect(url_for("routes.journalist_dashboard"))

@routes_bp.route("/journalist/download-private-key")
@journalist_required
def download_private_key():
    priv_pem = session.get("download_private_key") or session.get("active_private_key")
    if not priv_pem:
        flash("No private key available for download.", "warning")
        return redirect(url_for("routes.journalist_dashboard"))

    from io import BytesIO
    mem = BytesIO(priv_pem.encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"{session.get('username')}_private_key.pem", mimetype="application/x-pem-file")

@routes_bp.route("/journalist/sign-media", methods=["POST"])
@journalist_required
def journalist_sign_media():
    user_id = session["user_id"]
    if "media_file" not in request.files:
        flash("Please provide the original media file.", "danger")
        return redirect(url_for("routes.journalist_dashboard"))

    file = request.files["media_file"]
    if not file or not file.filename:
        flash("No media file selected.", "danger")
        return redirect(url_for("routes.journalist_dashboard"))

    # Validate file format and obtain media type ('image' or 'video')
    is_valid, err_msg, media_type = validate_uploaded_file(file, ip_address=request.remote_addr)
    if not is_valid:
        flash(err_msg, "danger")
        return redirect(url_for("routes.journalist_dashboard"))

    priv_key_file = request.files.get("private_key_file")
    use_session_key = request.form.get("use_session_key") == "true"
    
    priv_pem = None
    if use_session_key and session.get("active_private_key"):
        priv_pem = session.get("active_private_key")
    elif priv_key_file and priv_key_file.filename:
        priv_pem = priv_key_file.read().decode("utf-8", errors="ignore").strip()

    if not priv_pem or not priv_pem.startswith("-----BEGIN"):
        flash("Private key missing or invalid. Please upload your .pem file, or click 'Generate New RSA-2048 Keypair' on the left to create one.", "danger")
        return redirect(url_for("routes.journalist_dashboard"))

    title = request.form.get("title", "").strip() or file.filename
    description = request.form.get("description", "").strip()

    # Save file securely
    safe_name = sanitize_filename(file.filename)
    save_path = Path(current_app.config["UPLOAD_FOLDER"]) / safe_name
    file.save(str(save_path))

    # Calculate Hash, Perceptual Fingerprint & Digital Signature
    try:
        sha256_hex, sig_b64 = sign_media_file(save_path, priv_pem)
        p_hash = compute_media_fingerprint(save_path, media_type)
    except Exception as e:
        flash(f"Cryptographic signing failed: {e}", "danger")
        return redirect(url_for("routes.journalist_dashboard"))

    now_str = datetime.now().isoformat()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO media_records (
            uploader_id, filename, original_filename, file_type, file_size_bytes,
            file_path, sha256_hash, signature_b64, title, description, perceptual_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, safe_name, file.filename, media_type, save_path.stat().st_size,
        str(save_path), sha256_hex, sig_b64, title, description, p_hash, now_str
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    log_security_event(
        event_type="MEDIA_SIGNED",
        severity="INFO",
        description=f"Journalist signed original media: '{file.filename}' (SHA-256: {sha256_hex[:16]}...)",
        user_id=user_id,
        username=session.get("username"),
        ip_address=request.remote_addr
    )

    flash(f"Media successfully signed with RSA-PSS and registered to news wire! (Record #{record_id})", "success")
    return redirect(url_for("routes.journalist_dashboard"))

@routes_bp.route("/journalist/download-signature/<int:record_id>")
@journalist_required
def download_signature(record_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM media_records WHERE id = ?", (record_id,))
    rec = cursor.fetchone()
    conn.close()

    if not rec:
        flash("Record not found.", "danger")
        return redirect(url_for("routes.journalist_dashboard"))

    sig_payload = json.dumps({
        "original_filename": rec["original_filename"],
        "sha256_hash": rec["sha256_hash"],
        "signature_b64": rec["signature_b64"],
        "created_at": rec["created_at"],
        "algorithm": "RSA-PSS-SHA256"
    }, indent=2)

    from io import BytesIO
    mem = BytesIO(sig_payload.encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"{rec['original_filename']}.sig", mimetype="application/json")

# -------------------------------------------------------------
# 4. ADMINISTRATOR & SECURITY MONITORING DASHBOARD
# -------------------------------------------------------------

@routes_bp.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    conn = get_db()
    cursor = conn.cursor()

    # User Management
    cursor.execute("SELECT id, username, role, full_name, organization, is_approved, created_at FROM users ORDER BY id DESC")
    all_users = cursor.fetchall()

    # Threat Alerts
    cursor.execute("SELECT * FROM threat_alerts ORDER BY id DESC LIMIT 50")
    threat_alerts = cursor.fetchall()

    # Security Audit Logs
    cursor.execute("SELECT * FROM security_logs ORDER BY id DESC LIMIT 100")
    security_logs = cursor.fetchall()
    conn.close()

    return render_template(
        "admin_dashboard.html",
        users=all_users,
        threat_alerts=threat_alerts,
        security_logs=security_logs
    )

@routes_bp.route("/admin/approve-journalist/<int:user_id>", methods=["POST"])
@admin_required
def approve_journalist(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_approved = 1 WHERE id = ? AND role = 'journalist'", (user_id,))
    conn.commit()
    conn.close()

    log_security_event(
        event_type="JOURNALIST_APPROVED",
        severity="INFO",
        description=f"Administrator '{session.get('username')}' approved journalist ID: {user_id}",
        user_id=session.get("user_id"),
        username=session.get("username"),
        ip_address=request.remote_addr
    )
    flash(f"Journalist (ID #{user_id}) has been approved.", "success")
    return redirect(url_for("routes.admin_dashboard"))

@routes_bp.route("/admin/resolve-threat/<int:alert_id>", methods=["POST"])
@admin_required
def resolve_threat(alert_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE threat_alerts SET resolved = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()
    flash(f"Threat alert #{alert_id} marked as resolved.", "info")
    return redirect(url_for("routes.admin_dashboard"))
