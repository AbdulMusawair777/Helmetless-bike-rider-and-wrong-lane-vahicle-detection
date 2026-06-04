# ╔══════════════════════════════════════════════════════════════════════╗
# ║  app.py  —  Flask REST API + Dashboard Server                        ║
# ║  Integrates pipeline.py, database.py, report.py with the UI          ║
# ║  By Musaawar Khan                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# ── HOW TO RUN ────────────────────────────────────────────────────────
#   pip install flask flask-cors
#   python app.py
#   Then open: http://127.0.0.1:5000
# ══════════════════════════════════════════════════════════════════════

import os
import sys
import json
import uuid
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

# ── Import your existing modules ──────────────────────────────────────
import database as db
from report import export_reports, LiveCSVWriter

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
    from flask import render_template

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (snapshots, etc.)."""
    return send_from_directory('.', filename)


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