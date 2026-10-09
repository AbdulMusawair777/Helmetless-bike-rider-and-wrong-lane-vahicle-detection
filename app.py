import os
import sys
import json
import uuid
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file, Response
from flask_cors import CORS

# ── Import your existing modules ──────────────────────────────────────
import database as db
from report import export_reports, LiveCSVWriter
import alerts

# ══════════════════════════════════════════════════════════════════════
# FLASK APP SETUP
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='.')
CORS(app)   # allow dashboard.html to call the API from any origin

# ── Pipeline process state (one run at a time) ────────────────────────
pipeline_state = {
    'running':    False,
    'progress':   0,
    'status':     'idle',
    'log':        [],
    'session_id': None,
    'pid':        None,
}
_lock = threading.Lock()

# ── Ensure DB + output dirs exist ────────────────────────────────────
os.makedirs('output/snapshots', exist_ok=True)
os.makedirs('output/reports',   exist_ok=True)
db.init_db()


# ══════════════════════════════════════════════════════════════════════
# SERVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve dashboard.html as the main page."""
    return send_from_directory('.', 'dashboard.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (snapshots, etc.)."""
    return send_from_directory('.', filename)


# ══════════════════════════════════════════════════════════════════════
# LIVE VIDEO FEED — MJPEG stream of annotated frames
# pipeline.py writes output/latest_frame.jpg every processed frame;
# this watches that file and pushes it to the dashboard's <img> tag.
# ══════════════════════════════════════════════════════════════════════

_BOUNDARY = b'--frame'


class _FrameBuffer:
    """Thread-safe single-slot store for the latest JPEG frame bytes."""
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()
        self._new   = threading.Event()

    def push(self, data: bytes):
        with self._lock:
            self._frame = data
        self._new.set()
        self._new.clear()

    def get(self):
        with self._lock:
            return self._frame

    def wait(self, timeout=0.5):
        self._new.wait(timeout)


_frame_buffer = _FrameBuffer()
_LATEST_FRAME_PATH = os.path.join('output', 'latest_frame.jpg')


def _frame_watcher():
    """Polls output/latest_frame.jpg and feeds _frame_buffer at ~20 fps."""
    last_mtime = 0.0
    while True:
        try:
            if os.path.exists(_LATEST_FRAME_PATH):
                mt = os.path.getmtime(_LATEST_FRAME_PATH)
                if mt != last_mtime:
                    last_mtime = mt
                    with open(_LATEST_FRAME_PATH, 'rb') as f:
                        _frame_buffer.push(f.read())
        except Exception:
            pass
        time.sleep(0.05)


threading.Thread(target=_frame_watcher, daemon=True).start()

_PLACEHOLDER_JPEG = None


def _placeholder_jpeg() -> bytes:
    """A small 'waiting for pipeline...' JPEG so the <img> tag never
    shows a broken image while nothing has been streamed yet."""
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is not None:
        return _PLACEHOLDER_JPEG
    try:
        import cv2
        import numpy as np
        img = np.zeros((360, 640, 3), dtype='uint8')
        img[:] = (18, 24, 36)
        cv2.putText(img, 'Waiting for pipeline...', (140, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 90, 110), 1)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _PLACEHOLDER_JPEG = buf.tobytes() if ok else b''
    except Exception:
        _PLACEHOLDER_JPEG = b''
    return _PLACEHOLDER_JPEG


@app.route('/api/video_feed')
def video_feed():
    """
    MJPEG stream consumed by dashboard.html's <img id="video-feed">.
    - While the pipeline is running, serves frames pushed by _frame_watcher.
    - While idle, sends a placeholder frame every ~0.5s so the <img> tag
      doesn't render as a broken image.
    """
    def generate():
        while True:
            frame = _frame_buffer.get()
            if frame is None:
                frame = _placeholder_jpeg()
                time.sleep(0.5)
            else:
                _frame_buffer.wait(timeout=0.04)  # throttle to ~25 fps
            if not frame:
                time.sleep(0.5)
                continue
            yield (_BOUNDARY + b'\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


# ══════════════════════════════════════════════════════════════════════
# API — PIPELINE CONTROL
# ══════════════════════════════════════════════════════════════════════

def _run_pipeline_thread(video, direction, conf, iou, skip, ocr, preview,
                         moto_model, plate_model):
    """Runs pipeline.py in a subprocess, streams log lines into state."""
    global pipeline_state

    cmd = [
        sys.executable, 'pipeline.py',
        '--video', video,
        '--dir',   direction,
        '--conf',  str(conf),
        '--iou',   str(iou),
        '--skip',  str(skip),
        '--moto-model',  moto_model,
        '--plate-model', plate_model,
    ]
    if not ocr:
        cmd.append('--no-ocr')
    if preview:
        cmd.append('--preview')

    with _lock:
        pipeline_state['running']  = True
        pipeline_state['progress'] = 5
        pipeline_state['status']   = 'Starting pipeline...'
        pipeline_state['log']      = [f'$ {" ".join(cmd)}']

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with _lock:
            pipeline_state['pid'] = proc.pid

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with _lock:
                pipeline_state['log'].append(line)
                # Parse progress from pipeline's own print statements
                if '%]' in line:
                    try:
                        pct_str = line.strip().split('%]')[0].lstrip('[')
                        pct = float(pct_str)
                        pipeline_state['progress'] = int(pct)
                        pipeline_state['status']   = line.strip()
                    except Exception:
                        pass
                elif 'Loading' in line:
                    pipeline_state['progress'] = max(pipeline_state['progress'], 10)
                    pipeline_state['status']   = line.strip()
                elif 'Opening' in line or 'Initialising' in line:
                    pipeline_state['progress'] = max(pipeline_state['progress'], 20)
                    pipeline_state['status']   = line.strip()
                elif 'PROCESSING COMPLETE' in line:
                    pipeline_state['progress'] = 100
                    pipeline_state['status']   = 'Complete!'

        proc.wait()
        with _lock:
            pipeline_state['running']  = False
            pipeline_state['progress'] = 100 if proc.returncode == 0 else pipeline_state['progress']
            pipeline_state['status']   = ('Pipeline complete!' if proc.returncode == 0
                                          else f'Pipeline exited with code {proc.returncode}')
            pipeline_state['pid']      = None

    except Exception as e:
        with _lock:
            pipeline_state['running'] = False
            pipeline_state['status']  = f'Error: {e}'
            pipeline_state['log'].append(f'ERROR: {e}')


@app.route('/api/pipeline/start', methods=['POST'])
def pipeline_start():
    """POST /api/pipeline/start — launch the detection pipeline."""
    if pipeline_state['running']:
        return jsonify({'error': 'Pipeline is already running'}), 409

    data         = request.get_json() or {}
    video        = data.get('video',       'traffic.mp4')
    direction    = data.get('dir',         'right')
    conf         = float(data.get('conf',  0.35))
    iou          = float(data.get('iou',   0.45))
    skip         = int(data.get('skip',    1))
    ocr          = bool(data.get('ocr',    True))
    preview      = bool(data.get('preview',False))
    moto_model   = data.get('moto_model',  'models/motorcycle_best.pt')
    plate_model  = data.get('plate_model', 'models/plate_best.pt')

    if not os.path.exists(video):
        return jsonify({'error': f'Video file not found: {video}'}), 400
    if not os.path.exists(moto_model):
        return jsonify({'error': f'Model not found: {moto_model}'}), 400
    if not os.path.exists(plate_model):
        return jsonify({'error': f'Model not found: {plate_model}'}), 400

    # Clear any stale frame left over from a previous run
    try:
        if os.path.exists(_LATEST_FRAME_PATH):
            os.remove(_LATEST_FRAME_PATH)
    except Exception:
        pass

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(video, direction, conf, iou, skip, ocr, preview, moto_model, plate_model),
        daemon=True,
    )
    t.start()
    return jsonify({'message': 'Pipeline started', 'status': 'running'})


@app.route('/api/pipeline/status', methods=['GET'])
def pipeline_status():
    """GET /api/pipeline/status — poll progress, log, status."""
    with _lock:
        return jsonify({
            'running':  pipeline_state['running'],
            'progress': pipeline_state['progress'],
            'status':   pipeline_state['status'],
            'log':      pipeline_state['log'][-60:],  # last 60 lines
        })


@app.route('/api/pipeline/stop', methods=['POST'])
def pipeline_stop():
    """POST /api/pipeline/stop — kill the running pipeline subprocess."""
    with _lock:
        pid = pipeline_state.get('pid')
    if pid:
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
            return jsonify({'message': 'Stop signal sent'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'message': 'No pipeline running'})


# ══════════════════════════════════════════════════════════════════════
# API — VIOLATIONS
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/violations', methods=['GET'])
def get_violations():
    """GET /api/violations?session_id=...&type=WRONG_DIRECTION"""
    session_id = request.args.get('session_id')
    vtype      = request.args.get('type')
    rows = db.fetch_violations(session_id=session_id, violation_type=vtype)
    return jsonify(rows)


@app.route('/api/violations/summary', methods=['GET'])
def get_summary():
    """GET /api/violations/summary?session_id=..."""
    session_id = request.args.get('session_id')
    summary = db.fetch_summary(session_id=session_id)
    total   = sum(summary.values())
    return jsonify({
        'WRONG_DIRECTION': summary.get('WRONG_DIRECTION', 0),
        'NO_HELMET':       summary.get('NO_HELMET', 0),
        'BOTH':            summary.get('BOTH', 0),
        'TOTAL':           total,
    })


# ══════════════════════════════════════════════════════════════════════
# API — SESSIONS
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """GET /api/sessions — list all sessions newest first."""
    return jsonify(db.fetch_sessions())


@app.route('/api/sessions/latest', methods=['GET'])
def get_latest_session():
    """GET /api/sessions/latest — the most recent session."""
    sessions = db.fetch_sessions()
    if sessions:
        return jsonify(sessions[0])
    return jsonify(None)


# ══════════════════════════════════════════════════════════════════════
# API — PLATE LOOKUP
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/plates/search', methods=['GET'])
def plate_search():
    """GET /api/plates/search?q=ABC123"""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    return jsonify(db.plate_lookup(q))


@app.route('/api/plates/all', methods=['GET'])
def plates_all():
    """GET /api/plates/all — unique plate texts across all sessions."""
    session_id = request.args.get('session_id')
    rows = db.fetch_violations(session_id=session_id)
    plates = sorted({r['plate_text'] for r in rows if r.get('plate_text', '').strip()})
    return jsonify(plates)


# ══════════════════════════════════════════════════════════════════════
# API — REPORTS / EXPORT
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/reports/export', methods=['POST'])
def export():
    """POST /api/reports/export  body: {session_id: '...' | null}"""
    data       = request.get_json() or {}
    session_id = data.get('session_id')
    label      = (session_id[:16] if session_id else 'ALL')
    try:
        results = export_reports(session_id=session_id, label=label)
        return jsonify({
            'message': 'Export complete',
            'files': {k: v[0] for k, v in results.items()},
            'counts': {k: v[1] for k, v in results.items()},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/download/<path:filename>', methods=['GET'])
def download_report(filename):
    """GET /api/reports/download/<filename> — download a CSV."""
    safe = Path(filename).name
    path = os.path.join('output', 'reports', safe)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True)


# ══════════════════════════════════════════════════════════════════════
# API — SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/snapshots/<path:filename>', methods=['GET'])
def get_snapshot(filename):
    """GET /api/snapshots/<filename> — serve a violation JPEG."""
    return send_from_directory('output/snapshots', filename)


@app.route('/api/snapshots', methods=['GET'])
def list_snapshots():
    """GET /api/snapshots — list all snapshot filenames."""
    snap_dir = 'output/snapshots'
    if not os.path.isdir(snap_dir):
        return jsonify([])
    files = sorted(os.listdir(snap_dir))
    return jsonify(files)


# ══════════════════════════════════════════════════════════════════════
# API — ALERTS  (fine notifications via SMS / email)
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/alerts/config', methods=['GET'])
def alerts_config_get():
    """GET /api/alerts/config — current alert settings (secrets included,
    this app is assumed to run on a trusted local/LAN machine)."""
    return jsonify(alerts.load_config())


@app.route('/api/alerts/config', methods=['POST'])
def alerts_config_set():
    """POST /api/alerts/config — save alert settings (partial updates OK).
    Body mirrors alerts.DEFAULT_CONFIG, e.g.:
    {"enabled": true, "email": {"enabled": true, "gmail_user": "...",
     "gmail_app_password": "..."}, "sms": {...}, "fines": {...}}"""
    data = request.get_json() or {}
    saved = alerts.save_config(data)
    return jsonify(saved)


@app.route('/api/alerts/test', methods=['POST'])
def alerts_test():
    """POST /api/alerts/test  body: {email?, phone?} — sends a sample
    fine notice to verify credentials work."""
    data  = request.get_json() or {}
    email = data.get('email', '')
    phone = data.get('phone', '')
    if not email and not phone:
        return jsonify({'error': 'Provide an email and/or phone to test'}), 400
    cfg = alerts.load_config()
    result = alerts.send_test(cfg, email_to=email, phone_to=phone)
    return jsonify(result)


@app.route('/api/alerts/history', methods=['GET'])
def alerts_history():
    """GET /api/alerts/history?limit=200 — recent SMS/email send attempts."""
    limit = int(request.args.get('limit', 200))
    return jsonify(db.fetch_alerts(limit=limit))


@app.route('/api/alerts/resend/<int:violation_id>', methods=['POST'])
def alerts_resend(violation_id):
    """POST /api/alerts/resend/<id> — manually (re)send the fine notice
    for one specific violation row, plate lookup + snapshot attachment
    included, and return the result immediately (not fire-and-forget)."""
    row = db.fetch_violation_by_id(violation_id)
    if not row:
        return jsonify({'error': 'Violation not found'}), 404
    result = alerts.resend_alert(row)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════
# API — VEHICLE OWNERS  (plate → contact for fine notices)
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/owners', methods=['GET'])
def owners_list():
    """GET /api/owners — every registered plate → owner contact."""
    return jsonify(db.list_owners())


@app.route('/api/owners', methods=['POST'])
def owners_upsert():
    """POST /api/owners  body: {plate, owner_name?, phone?, email?}
    Registers or updates the owner contact for a plate. `phone` should
    be in E.164 format, e.g. +923001234567, for SMS delivery."""
    data  = request.get_json() or {}
    plate = (data.get('plate') or '').strip()
    if not plate:
        return jsonify({'error': 'plate is required'}), 400
    owner = db.upsert_owner(
        plate_text = plate,
        owner_name = data.get('owner_name', ''),
        phone      = data.get('phone', ''),
        email      = data.get('email', ''),
    )
    return jsonify(owner)


@app.route('/api/owners/<path:plate>', methods=['DELETE'])
def owners_delete(plate):
    """DELETE /api/owners/<plate> — remove a registered owner."""
    db.delete_owner(plate)
    return jsonify({'message': 'deleted'})


# ══════════════════════════════════════════════════════════════════════
# API — SYSTEM INFO
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/system', methods=['GET'])
def system_info():
    """GET /api/system — check model files and output folder."""
    return jsonify({
        'moto_model_exists':  os.path.exists('models/motorcycle_best.pt'),
        'plate_model_exists': os.path.exists('models/plate_best.pt'),
        'db_exists':          os.path.exists('output/violations.db'),
        'output_dir_exists':  os.path.isdir('output'),
        'python':             sys.version,
        'server_time':        datetime.now().isoformat(sep=' ', timespec='seconds'),
    })


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('═' * 60)
    print('  ViolationIQ  —  Dashboard Server')
    print('═' * 60)
    print('  Open in browser:  http://127.0.0.1:5000')
    print('  API base:         http://127.0.0.1:5000/api')
    print('═' * 60)
    print()
    app.run(host='0.0.0.0', port=5000, debug=False)