import os
import json
import smtplib
import ssl
import threading
import mimetypes
from email.message import EmailMessage
from datetime import datetime

import database as db

CONFIG_PATH = os.path.join('output', 'alert_config.json')

# ── Fine schedule (edit freely — amounts are in PKR by default) ───────
DEFAULT_FINES = {
    'WRONG_DIRECTION': 2000,
    'NO_HELMET':        1000,
    'BOTH':              2500,
}

DEFAULT_CONFIG = {
    'enabled': False,
    'currency': 'PKR',
    'fines': DEFAULT_FINES,
    'email': {
        'enabled':       False,
        'smtp_host':     'smtp.gmail.com',
        'smtp_port':     465,
        'gmail_user':    '',
        'gmail_app_password': '',
    },
    'sms': {
        'enabled':      False,
        'twilio_sid':   '',
        'twilio_token': '',
        'twilio_from':  '',
    },
    # Fallback recipient (e.g. traffic-cell inbox/phone) used when a
    # plate has no registered owner, so nothing is silently dropped.
    'fallback_email': '',
    'fallback_phone': '',
}


# ─────────────────────────────────────────────────────────────────────
# CONFIG LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # backfill any missing keys from defaults (new fields, etc.)
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        _deep_merge(merged, cfg)
        return merged
    except Exception as e:
        print(f'  [ALERTS] Failed to read config, using defaults: {e}')
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    merged = load_config()
    _deep_merge(merged, cfg)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)
    return merged


def _deep_merge(base: dict, incoming: dict):
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ─────────────────────────────────────────────────────────────────────
# MESSAGE TEXT
# ─────────────────────────────────────────────────────────────────────

def _fine_amount(cfg: dict, vtype: str) -> int:
    return int(cfg.get('fines', DEFAULT_FINES).get(vtype, 1000))


def _violation_label(vtype: str) -> str:
    return {
        'WRONG_DIRECTION': 'Wrong-direction driving',
        'NO_HELMET':       'Riding without a helmet',
        'BOTH':             'Wrong-direction driving + no helmet',
    }.get(vtype, vtype)


def build_message(cfg: dict, violation: dict, owner_name: str = '') -> tuple:
    """Returns (subject, body) for a fine notification."""
    vtype   = violation['violation_type']
    fine    = _fine_amount(cfg, vtype)
    curr    = cfg.get('currency', 'PKR')
    plate   = violation.get('plate_text') or 'UNKNOWN'
    when    = violation.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    label   = _violation_label(vtype)
    hello   = f'Dear {owner_name},' if owner_name else 'Dear Vehicle Owner,'

    subject = f'Traffic Violation Notice — Plate {plate}'
    body = (
        f'{hello}\n\n'
        f'A traffic violation was recorded against your vehicle.\n\n'
        f'  Violation   : {label}\n'
        f'  Plate       : {plate}\n'
        f'  Date/Time   : {when}\n'
        f'  Fine Amount : {curr} {fine:,}\n\n'
        f'Please settle this fine within 14 days to avoid additional '
        f'penalties. If you believe this notice was issued in error, '
        f'contact the traffic authority with the plate number above.\n\n'
        f'— Automated Traffic Violation Detection System'
    )
    return subject, body


def build_sms_text(cfg: dict, violation: dict) -> str:
    vtype = violation['violation_type']
    fine  = _fine_amount(cfg, vtype)
    curr  = cfg.get('currency', 'PKR')
    plate = violation.get('plate_text') or 'UNKNOWN'
    label = _violation_label(vtype)
    return (f'Traffic Notice: Plate {plate} — {label}. '
            f'Fine: {curr} {fine:,}. Pay within 14 days.')


# ─────────────────────────────────────────────────────────────────────
# SENDERS
# ─────────────────────────────────────────────────────────────────────

def send_email(cfg: dict, to_addr: str, subject: str, body: str,
               attachment_path: str = '') -> tuple:
    """Returns (ok: bool, info: str). If attachment_path points to a
    real, readable image file, it is attached to the email (the
    violation snapshot showing the plate/rider)."""
    ecfg = cfg.get('email', {})
    if not ecfg.get('enabled'):
        return False, 'email channel disabled'
    user = ecfg.get('gmail_user', '')
    pwd  = ecfg.get('gmail_app_password', '')
    if not user or not pwd or not to_addr:
        return False, 'missing gmail credentials or recipient'

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From']    = user
    msg['To']      = to_addr
    msg.set_content(body)

    if attachment_path and os.path.isfile(attachment_path):
        try:
            ctype, _ = mimetypes.guess_type(attachment_path)
            ctype = ctype or 'image/jpeg'
            maintype, subtype = ctype.split('/', 1)
            with open(attachment_path, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                                    filename=os.path.basename(attachment_path))
        except Exception:
            pass  # send without the attachment rather than fail the whole alert

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(ecfg.get('smtp_host', 'smtp.gmail.com'),
                               int(ecfg.get('smtp_port', 465)),
                               context=ctx) as server:
            server.login(user, pwd)
            server.send_message(msg)
        return True, 'sent'
    except Exception as e:
        return False, str(e)


def send_sms(cfg: dict, to_number: str, text: str) -> tuple:
    """Returns (ok: bool, info: str). Requires `pip install twilio`."""
    scfg = cfg.get('sms', {})
    if not scfg.get('enabled'):
        return False, 'sms channel disabled'
    sid   = scfg.get('twilio_sid', '')
    token = scfg.get('twilio_token', '')
    from_ = scfg.get('twilio_from', '')
    if not sid or not token or not from_ or not to_number:
        return False, 'missing twilio credentials or recipient'

    try:
        from twilio.rest import Client
    except ImportError:
        return False, 'twilio package not installed (pip install twilio)'

    try:
        client = Client(sid, token)
        client.messages.create(body=text, from_=from_, to=to_number)
        return True, 'sent'
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────
# DISPATCH  (called from pipeline.py — always in a background thread)
# ─────────────────────────────────────────────────────────────────────

def _dispatch_sync(violation: dict):
    """
    Look up the owner by plate, build the notice, send email + SMS,
    and log every attempt (success or failure) to the alerts_log table.
    """
    cfg = load_config()
    if not cfg.get('enabled'):
        return

    plate = (violation.get('plate_text') or '').strip().upper()
    owner = db.get_owner(plate) if plate else None

    email_to = (owner or {}).get('email') or cfg.get('fallback_email', '')
    phone_to = (owner or {}).get('phone') or cfg.get('fallback_phone', '')
    owner_nm = (owner or {}).get('owner_name', '')

    subject, body = build_message(cfg, violation, owner_nm)
    sms_text       = build_sms_text(cfg, violation)
    snapshot       = violation.get('snapshot_path', '')

    if email_to:
        ok, info = send_email(cfg, email_to, subject, body, attachment_path=snapshot)
        db.log_alert(violation.get('id'), plate, 'email', email_to,
                      'sent' if ok else 'failed', info)
    else:
        db.log_alert(violation.get('id'), plate, 'email', '',
                      'skipped', 'no owner email on file')

    if phone_to:
        ok, info = send_sms(cfg, phone_to, sms_text)
        db.log_alert(violation.get('id'), plate, 'sms', phone_to,
                      'sent' if ok else 'failed', info)
    else:
        db.log_alert(violation.get('id'), plate, 'sms', '',
                      'skipped', 'no owner phone on file')


def dispatch_alert(violation: dict):
    """Fire-and-forget: never blocks the caller (frame loop)."""
    t = threading.Thread(target=_dispatch_sync, args=(violation,), daemon=True)
    t.start()


def resend_alert(violation: dict) -> dict:
    """
    Synchronous version of _dispatch_sync used by the dashboard's manual
    "Resend alert" button, so the UI can show the result right away
    instead of polling the alert history table.
    """
    cfg = load_config()
    if not cfg.get('enabled'):
        return {'error': 'Alerts are disabled — turn on the master switch first'}

    plate = (violation.get('plate_text') or '').strip().upper()
    owner = db.get_owner(plate) if plate else None

    email_to = (owner or {}).get('email') or cfg.get('fallback_email', '')
    phone_to = (owner or {}).get('phone') or cfg.get('fallback_phone', '')
    owner_nm = (owner or {}).get('owner_name', '')

    subject, body = build_message(cfg, violation, owner_nm)
    sms_text       = build_sms_text(cfg, violation)
    snapshot       = violation.get('snapshot_path', '')

    result = {'owner_found': owner is not None, 'sent_to_email': email_to, 'sent_to_phone': phone_to}

    if email_to:
        ok, info = send_email(cfg, email_to, subject, body, attachment_path=snapshot)
        db.log_alert(violation.get('id'), plate, 'email', email_to,
                      'sent' if ok else 'failed', info)
        result['email'] = {'ok': ok, 'info': info}
    else:
        result['email'] = {'ok': False, 'info': 'no recipient email on file'}

    if phone_to:
        ok, info = send_sms(cfg, phone_to, sms_text)
        db.log_alert(violation.get('id'), plate, 'sms', phone_to,
                      'sent' if ok else 'failed', info)
        result['sms'] = {'ok': ok, 'info': info}
    else:
        result['sms'] = {'ok': False, 'info': 'no recipient phone on file'}

    return result


def send_test(cfg: dict, email_to: str = '', phone_to: str = '') -> dict:
    """Used by the dashboard 'Send test alert' button."""
    result = {}
    fake_violation = {
        'id': -1,
        'violation_type': 'NO_HELMET',
        'plate_text': 'TEST-1234',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    subject, body = build_message(cfg, fake_violation, 'Test User')
    sms_text       = build_sms_text(cfg, fake_violation)

    if email_to:
        ok, info = send_email(cfg, email_to, subject, body)
        result['email'] = {'ok': ok, 'info': info}
    if phone_to:
        ok, info = send_sms(cfg, phone_to, sms_text)
        result['sms'] = {'ok': ok, 'info': info}
    return result