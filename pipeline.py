# ╔══════════════════════════════════════════════════════════════════════╗
# ║  pipeline.py  —  Video-Based Violation Detection System              ║
# ║  By Musaawar Khan                                                    ║
# ║                                                                      ║
# ║  INPUT  : any video file  (.mp4 / .avi / .mov / .mkv)               ║
# ║  OUTPUT :                                                            ║
# ║    output/result_<videoname>.mp4   ← annotated video                ║
# ║    output/violations.db            ← SQLite database                ║
# ║    output/reports/<session>/       ← 4 live CSV files               ║
# ║    output/snapshots/               ← violation snapshot JPEGs       ║
# ║                                                                      ║
# ║  Models                              ║
# ║    models/motorcycle_best.pt  →  helmet / helmetless / motorcycle   ║
# ║    models/plate_best.pt       →  license plate region               ║
# ║    EasyOCR                    →  reads plate number text            ║
# ║                                                                      ║
# ║  Flags written to video, DB, and CSV:                                ║
# ║    WRONG_DIRECTION  — vehicle going the wrong way                    ║
# ║    NO_HELMET        — helmetless rider                               ║
# ║    BOTH             — wrong direction + no helmet                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# ── SETUP (run once) ──────────────────────────────────────────────────
#   pip install -r requirements.txt
#
# ── FOLDER STRUCTURE ──────────────────────────────────────────────────
#   project/
#   ├── pipeline.py
#   ├── database.py
#   ├── report.py
#   ├── requirements.txt
#   ├── models/
#   │   ├── motorcycle_best.pt
#   │   └── plate_best.pt
#   └── output/                        (auto-created)
#
# ── HOW TO RUN ────────────────────────────────────────────────────────
#
#   Basic (vehicles should travel LEFT → RIGHT):
#     python pipeline.py --video traffic.mp4
#
#   Road where traffic travels RIGHT → LEFT:
#     python pipeline.py --video traffic.mp4 --dir left
#
#   Skip OCR for faster processing:
#     python pipeline.py --video traffic.mp4 --no-ocr
#
#   Process every 2nd frame (faster on long videos):
#     python pipeline.py --video traffic.mp4 --skip 2
#
#   Show preview window while processing:
#     python pipeline.py --video traffic.mp4 --preview
#
#   Full example with all options:
#     python pipeline.py \
#       --video  traffic.mp4 \
#       --dir    right \
#       --conf   0.35 \
#       --skip   1 \
#       --preview
#
# ── AFTER THE RUN ─────────────────────────────────────────────────────
#   python report.py              ← re-export CSVs from latest session
#   python report.py --list       ← list all sessions in DB
#   python report.py --plate XYZ  ← plate number lookup
# ══════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import easyocr
import time
import os
import sys
import argparse
import uuid
from datetime import datetime
from pathlib import Path

import database as db
from report import LiveCSVWriter


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description='Video-Based Vehicle Violation Detector',
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        '--video', required=True,
        help='Path to the input video file\n'
             'Example: --video traffic.mp4'
    )
    p.add_argument(
        '--dir', default='right', choices=['right', 'left'],
        help='Legal travel direction on this road\n'
             'right = vehicles should move left → right  (default)\n'
             'left  = vehicles should move right → left'
    )
    p.add_argument(
        '--conf', default=0.35, type=float,
        help='Detection confidence threshold  (default: 0.35)'
    )
    p.add_argument(
        '--iou', default=0.45, type=float,
        help='NMS IoU threshold  (default: 0.45)'
    )
    p.add_argument(
        '--skip', default=1, type=int,
        help='Process every Nth frame  (default: 1 = every frame)\n'
             'Use 2 or 3 for faster processing on long videos'
    )
    p.add_argument(
        '--no-ocr', action='store_true',
        help='Disable EasyOCR plate reading (faster, no plate text)'
    )
    p.add_argument(
        '--preview', action='store_true',
        help='Show a preview window while processing\n'
             'Press Q to stop early'
    )
    p.add_argument(
        '--moto-model',  default='models/motorcycle_best.pt',
        help='Path to motorcycle_best.pt  (default: models/motorcycle_best.pt)'
    )
    p.add_argument(
        '--plate-model', default='models/plate_best.pt',
        help='Path to plate_best.pt  (default: models/plate_best.pt)'
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# VIOLATION TYPES
# ══════════════════════════════════════════════════════════════════════

VT_WRONG = 'WRONG_DIRECTION'
VT_HELM  = 'NO_HELMET'
VT_BOTH  = 'BOTH'


# ══════════════════════════════════════════════════════════════════════
# COLOURS  (BGR)
# ══════════════════════════════════════════════════════════════════════

C = {
    'correct':  (50,  220, 100),   # green
    'wrong':    (0,   50,  255),   # red
    'helmet':   (0,   200,  80),   # green
    'nohelm':   (0,   50,  255),   # red
    'plate':    (0,   210, 255),   # cyan
    'roi':      (255, 170,  30),   # amber
    'tracking': (160, 160, 160),   # grey (not yet flagged)
    'white':    (255, 255, 255),
    'dark':     (15,  15,  15),
}
FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX


# ══════════════════════════════════════════════════════════════════════
# DIRECTION TRACKER
# ══════════════════════════════════════════════════════════════════════

class DirectionTracker:
    """
    Tracks centroid history per vehicle and determines travel direction.

    correct_dir = +1  →  legal direction is left → right (positive dx)
    correct_dir = -1  →  legal direction is right → left (negative dx)

    Returns 'correct' | 'wrong' | 'unknown'
    """

    def __init__(self, correct_dir: int = 1,
                 history_len: int = 32,
                 min_frames:  int = 8,
                 min_px:      int = 18):
        self.correct_dir  = correct_dir
        self.min_frames   = min_frames
        self.min_px       = min_px
        self.histories    = defaultdict(lambda: deque(maxlen=history_len))
        self.directions   = {}
        self.frame_counts = defaultdict(int)

    def update(self, tid: int, cx: int, cy: int) -> str:
        self.histories[tid].append((cx, cy))
        self.frame_counts[tid] += 1
        h = self.histories[tid]
        if len(h) < self.min_frames:
            return self.directions.get(tid, 'unknown')
        dx = h[-1][0] - h[0][0]
        if abs(dx) < self.min_px:
            return self.directions.get(tid, 'unknown')
        label = 'correct' if (dx * self.correct_dir) > 0 else 'wrong'
        self.directions[tid] = label
        return label

    def cleanup(self, active: set):
        stale = set(self.histories) - active
        for tid in stale:
            self.histories.pop(tid, None)
            self.directions.pop(tid, None)
            self.frame_counts.pop(tid, None)


# ══════════════════════════════════════════════════════════════════════
# MOTORCYCLE + HELMET DETECTOR
# ══════════════════════════════════════════════════════════════════════

class MotoDetector:
    """
    Runs motorcycle_best.pt on the full frame.

    Class name matching (case-insensitive, partial):
      contains 'no' + 'helmet'  OR  'helmetless'  →  NO HELMET
      contains 'helmet'  (without 'no'/'without')  →  Helmet
      anything else                                →  Motorcycle (generic)

    If your model uses different class names, adjust _is_helmet
    and _is_nohelmet below.
    """

    def __init__(self, model_path: str, conf: float = 0.30, iou: float = 0.45):
        if not os.path.exists(model_path):
            print(f'\n  ERROR: motorcycle model not found: {model_path}')
            print(f'  Train it on Colab (train_motorcycle_colab.py) and place')
            print(f'  the downloaded best.pt at: {model_path}\n')
            sys.exit(1)
        self.model = YOLO(model_path)
        self.conf  = conf
        self.iou   = iou
        self.names = self.model.names
        print(f'  [Moto]  Loaded → {model_path}')
        print(f'  [Moto]  Classes: {list(self.names.values())}')

    @staticmethod
    def _is_helmet(n: str) -> bool:
        n = n.lower()
        return 'helmet' in n and 'no' not in n and 'without' not in n

    @staticmethod
    def _is_nohelmet(n: str) -> bool:
        n = n.lower()
        return (('no' in n and 'helmet' in n)
                or 'without' in n
                or 'helmetless' in n)

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns list of dicts:
          x1, y1, x2, y2   — bounding box
          cls_name          — raw class name from YOUR model
          helmet_status     — 'Helmet' | 'NO HELMET' | 'Motorcycle'
          conf              — detection confidence
        """
        res = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = self.names[int(box.cls[0])]
            h_status = ('NO HELMET' if self._is_nohelmet(cls_name) else
                        'Helmet'    if self._is_helmet(cls_name)    else
                        'Motorcycle')
            out.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cls_name':      cls_name,
                'helmet_status': h_status,
                'conf':          float(box.conf[0]),
            })
        return out


# ══════════════════════════════════════════════════════════════════════
# LICENSE PLATE DETECTOR + OCR
# ══════════════════════════════════════════════════════════════════════

class PlateDetector:
    """
    Runs plate_best.pt on the cropped vehicle region to find the plate box.
    Then uses EasyOCR to read the plate number.

    Returns full-frame plate coordinates so the box can be drawn correctly.
    """

    def __init__(self, model_path: str, conf: float = 0.25,
                 use_ocr: bool = True, gpu: bool = False):
        if not os.path.exists(model_path):
            print(f'\n  ERROR: plate model not found: {model_path}')
            print(f'  Train it on Colab (train_plate_colab.py) and place')
            print(f'  the downloaded best.pt at: {model_path}\n')
            sys.exit(1)
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.use_ocr = use_ocr
        self.ocr     = None
        print(f'  [Plate] Loaded → {model_path}')
        print(f'  [Plate] Classes: {list(self.model.names.values())}')
        if use_ocr:
            print('  [OCR]   Initialising EasyOCR...')
            self.ocr = easyocr.Reader(['en'], gpu=gpu, verbose=False)
            print('  [OCR]   Ready.')

    def detect(self, frame: np.ndarray,
               vx1: int, vy1: int, vx2: int, vy2: int) -> list:
        """
        Crop the vehicle region, run plate model, OCR each plate found.

        Returns list of dicts:
          px1, py1, px2, py2  — plate box in FULL-FRAME coordinates
          plate_text          — OCR string ('' if unreadable or OCR off)
          conf                — plate detection confidence
        """
        crop = frame[max(vy1, 0):vy2, max(vx1, 0):vx2]
        if crop.size == 0:
            return []

        res    = self.model(crop, conf=self.conf, iou=0.40, verbose=False)[0]
        plates = []

        for box in res.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            # Convert crop-relative → full-frame coordinates
            fx1 = vx1 + bx1;  fy1 = vy1 + by1
            fx2 = vx1 + bx2;  fy2 = vy1 + by2

            text = ''
            if self.ocr is not None:
                plate_crop = frame[max(fy1, 0):fy2, max(fx1, 0):fx2]
                if plate_crop.size > 0:
                    # Upscale tiny plates → better OCR accuracy
                    ph, pw = plate_crop.shape[:2]
                    if pw > 0 and pw < 120:
                        scale      = 120 / pw
                        plate_crop = cv2.resize(
                            plate_crop,
                            (int(pw * scale), int(ph * scale)),
                            interpolation=cv2.INTER_CUBIC
                        )
                    ocr_res = self.ocr.readtext(
                        plate_crop, detail=1,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '
                    )
                    text = ' '.join(
                        r[1] for r in ocr_res if r[2] > 0.20
                    ).strip().upper()

            plates.append({
                'px1': fx1, 'py1': fy1,
                'px2': fx2, 'py2': fy2,
                'plate_text': text,
                'conf': float(box.conf[0]),
            })

        return plates


# ══════════════════════════════════════════════════════════════════════
# PLATE MEMORY
# Stores the best (longest non-empty) OCR read per track_id.
# ══════════════════════════════════════════════════════════════════════

class PlateMemory:
    def __init__(self):
        self._best: dict[int, str] = {}
        # last known plate box per tid for persistent display
        self._box:  dict[int, tuple] = {}

    def update(self, tid: int, text: str,
               px1=0, py1=0, px2=0, py2=0) -> tuple:
        """
        Returns (improved: bool, best_text: str).
        improved=True  →  we have a better reading, callers should patch DB/CSV.
        """
        new = text.upper().strip()
        old = self._best.get(tid, '')
        if px1 or py1:
            self._box[tid] = (px1, py1, px2, py2)
        if new and len(new) > len(old):
            self._best[tid] = new
            return True, new
        return False, old

    def get_text(self, tid: int) -> str:
        return self._best.get(tid, '')

    def get_box(self, tid: int) -> tuple:
        return self._box.get(tid, ())

    def count_read(self) -> int:
        return sum(1 for v in self._best.values() if v)


# ══════════════════════════════════════════════════════════════════════
# VIOLATION RECORDER
# ══════════════════════════════════════════════════════════════════════

class ViolationRecorder:
    """
    Writes one DB row + one CSV row per new (track_id × violation_type).
    Saves a snapshot JPEG on first occurrence.
    On a later frame when OCR finally reads a plate, patches all
    existing DB rows and CSV rows for that track_id.
    """

    def __init__(self, session_id, video_source, fps, snapshot_dir, csv_writer):
        self.session_id   = session_id
        self.video_source = video_source
        self.fps          = max(fps, 1.0)
        self.snap_dir     = snapshot_dir
        self.csv          = csv_writer
        self.snapped      = set()              # (tid, vtype) already recorded
        self.count        = 0
        self.row_ids      = defaultdict(list)  # tid → [db row ids]

    def record(self, frame, frame_no, tid, vehicle_class,
               vtype, direction, helmet_status,
               plate_detected, plate_text,
               bbox, confidence) -> bool:
        """
        Write to DB + CSV on FIRST occurrence only.
        Returns True on first occurrence, False (no-op) on repeat.

        Each (track_id, violation_type) pair is stored exactly once per
        session.  Subsequent frames that confirm the same violation are
        silently skipped — only plate upgrades (via patch_plate) can
        update an existing row after it is written.
        """
        key = (tid, vtype)

        # ── Already recorded this (track, violation) — skip entirely ──
        if key in self.snapped:
            return False

        # ── First occurrence: mark, snapshot, write ────────────────────
        self.snapped.add(key)

        fname     = f'{vtype.lower()}_id{tid}_f{frame_no}.jpg'
        snap_path = os.path.join(self.snap_dir, fname)
        cv2.imwrite(snap_path, frame)

        # Write one row to SQLite
        row_id = db.insert_violation(
            session_id     = self.session_id,
            video_source   = self.video_source,
            frame_number   = frame_no,
            frame_time_sec = frame_no / self.fps,
            track_id       = tid,
            vehicle_class  = vehicle_class,
            violation_type = vtype,
            direction      = direction,
            helmet_status  = helmet_status,
            plate_detected = plate_detected,
            plate_text     = plate_text,
            snapshot_path  = snap_path,
            confidence     = confidence,
            bbox           = bbox,
        )
        self.row_ids[tid].append(row_id)

        # Write one row to live CSV
        x1, y1, x2, y2 = bbox
        self.csv.write({
            'id':              row_id,
            'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_id':      self.session_id,
            'video_source':    self.video_source,
            'frame_number':    frame_no,
            'frame_time_sec':  round(frame_no / self.fps, 2),
            'track_id':        tid,
            'vehicle_class':   vehicle_class,
            'violation_type':  vtype,
            'direction':       direction,
            'helmet_status':   helmet_status,
            'plate_detected':  int(plate_detected),
            'plate_text':      plate_text.upper().strip(),
            'plate_updated_at': '',
            'snapshot_path':   snap_path,
            'confidence':      round(confidence, 4),
            'bbox_x1': x1, 'bbox_y1': y1,
            'bbox_x2': x2, 'bbox_y2': y2,
            'reviewed': 0, 'notes': '',
        }, vtype)
        self.count += 1
        return True

    def patch_plate(self, tid: int, plate_text: str):
        """Retrofit plate_text into DB + CSV when OCR improves."""
        pt = plate_text.upper().strip()
        if not pt:
            return
        db.update_plate_text(self.session_id, tid, pt)
        self.csv.update_plate(tid, pt)


# ══════════════════════════════════════════════════════════════════════
# DRAWING  —  all annotation rendered onto the output video frame
# ══════════════════════════════════════════════════════════════════════

def draw_trail(frame, history, color):
    """Direction trail arrow following the vehicle."""
    pts = list(history)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        cv2.line(frame, pts[i-1], pts[i],
                 tuple(int(ch * alpha) for ch in color), 2, cv2.LINE_AA)
    if len(pts) >= 4:
        cv2.arrowedLine(frame, pts[-4], pts[-1],
                        color, 2, cv2.LINE_AA, tipLength=0.45)


def draw_vehicle_box(frame, x1, y1, x2, y2, color, label, thick=2):
    """Vehicle bounding box with corner ticks and top label badge."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    cl = 14
    for cx, cy, dx, dy in [
        (x1, y1, 1,  1), (x2, y1, -1,  1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*cl, cy), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy*cl), color, 3, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.52, 1)
    by1 = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, by1), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 4),
                FONT, 0.52, C['white'], 1, cv2.LINE_AA)


def draw_tag(frame, x, y, text, bg_color):
    """Small filled tag (helmet status, etc.) below the vehicle box."""
    (tw, th), _ = cv2.getTextSize(text, FONTS, 0.46, 1)
    cv2.rectangle(frame, (x, y), (x + tw + 8, y + th + 6), bg_color, -1)
    cv2.putText(frame, text, (x + 4, y + th + 2),
                FONTS, 0.46, C['white'], 1, cv2.LINE_AA)


def draw_plate(frame, px1, py1, px2, py2, plate_text: str):
    """
    Draws the plate bounding box in cyan and shows the OCR text above it.

    Always drawn whenever plate_best.pt fires — with or without a flag.

    If plate_text is empty the badge says 'Reading plate...' so the operator
    can see the system detected a plate even when OCR hasn't read it yet.
    """
    # Plate rectangle
    cv2.rectangle(frame, (px1, py1), (px2, py2), C['plate'], 2, cv2.LINE_AA)
    tl = 10
    for cx, cy, dx, dy in [
        (px1, py1, 1,  1), (px2, py1, -1,  1),
        (px1, py2, 1, -1), (px2, py2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*tl, cy), C['plate'], 3, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy*tl), C['plate'], 3, cv2.LINE_AA)

    # Text badge above the plate box
    badge = plate_text if plate_text else 'Reading plate...'
    scale = 0.65
    thick = 2
    (tw, th), _ = cv2.getTextSize(badge, FONT, scale, thick)
    pad  = 5
    bx1  = px1
    by2  = max(py1 - 2, th + pad * 2)
    by1  = by2 - th - pad * 2

    cv2.rectangle(frame, (bx1, by1), (bx1 + tw + pad*2, by2), C['dark'], -1)
    cv2.rectangle(frame, (bx1, by1), (bx1 + tw + pad*2, by2), C['plate'], 1)
    cv2.putText(frame, badge, (bx1 + pad, by2 - pad),
                FONT, scale, C['plate'], thick, cv2.LINE_AA)


def draw_roi(frame, poly, alpha=0.08):
    """Semi-transparent ROI lane overlay."""
    if poly is None:
        return
    ov = frame.copy()
    cv2.fillPoly(ov, [poly], C['roi'])
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [poly], True, C['roi'], 2, cv2.LINE_AA)


def draw_hud(frame, fps, n_wrong, n_helm, n_total,
             dir_str, frame_no, total_frames, plates_read):
    """
    Info panel in the top-left corner showing live stats.
    Includes a progress bar for the video.
    """
    lines = [
        f'  FPS           : {fps:5.1f}',
        f'  Correct Dir   : {dir_str.upper()}',
        f'  Vehicles      : {n_total}',
        f'  Wrong Dir     : {n_wrong}',
        f'  No Helmet     : {n_helm}',
        f'  Plates Read   : {plates_read}',
        f'  Frame         : {frame_no}/{total_frames or "?"}',
    ]
    lh, pad, w = 22, 8, 270
    h = len(lines) * lh + pad * 2 + 16   # +16 for progress bar

    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (10 + w, 10 + h), C['dark'], -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + w, 10 + h), (80, 80, 80), 1)

    for i, line in enumerate(lines):
        col = (C['wrong']   if 'Wrong'  in line and n_wrong > 0   else
               C['nohelm']  if 'Helmet' in line and n_helm > 0    else
               C['plate']   if 'Plates' in line and plates_read > 0 else
               (80, 255, 130) if 'FPS'  in line else
               C['white'])
        cv2.putText(frame, line,
                    (14, 10 + pad + (i + 1) * lh - 4),
                    FONTS, 0.49, col, 1, cv2.LINE_AA)

    # Progress bar
    if total_frames > 0:
        bar_x1   = 14
        bar_y    = 10 + pad + len(lines) * lh + 6
        bar_w    = w - 8
        bar_h    = 8
        progress = int(bar_w * frame_no / total_frames)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + progress, bar_y + bar_h),
                      (80, 255, 130), -1)


def draw_alert_banner(frame, wrong_ids, helm_ids):
    """Red alert banner at the bottom of the frame."""
    alerts = []
    if wrong_ids:
        ids = ', '.join(str(i) for i in sorted(wrong_ids)[:6])
        alerts.append(f'  WRONG DIRECTION  |  IDs: {ids}')
    if helm_ids:
        ids = ', '.join(str(i) for i in sorted(helm_ids)[:6])
        alerts.append(f'  NO HELMET RIDER  |  IDs: {ids}')
    if not alerts:
        return
    fh, fw = frame.shape[:2]
    bh     = 38 * len(alerts) + 8
    ov     = frame.copy()
    cv2.rectangle(ov, (0, fh - bh), (fw, fh), (0, 0, 160), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    for i, msg in enumerate(alerts):
        cv2.putText(frame, msg, (10, fh - bh + 28 + i * 38),
                    FONT, 0.62, C['white'], 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
# ROI  (lane region of interest)
# ══════════════════════════════════════════════════════════════════════

def build_roi(W: int, H: int) -> np.ndarray:
    """Default lane band: 25%–80% frame height, 2%–98% width."""
    return np.array([
        [int(W * 0.02), int(H * 0.25)],
        [int(W * 0.98), int(H * 0.25)],
        [int(W * 0.98), int(H * 0.80)],
        [int(W * 0.02), int(H * 0.80)],
    ], dtype=np.int32)


def in_roi(cx: int, cy: int, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0


# ══════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR  —  the main pipeline
# ══════════════════════════════════════════════════════════════════════

def process_video(args):
    # ── Validate input video ──────────────────────────────────
    video_path = args.video
    if not os.path.isfile(video_path):
        print(f'\n  ERROR: Video file not found: {video_path}')
        print(f'  Check the path and try again.\n')
        sys.exit(1)

    video_name   = Path(video_path).stem
    session_id   = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output_video = os.path.join('output', f'result_{video_name}.mp4')

    print('\n' + '═'*64)
    print('  VIDEO VIOLATION DETECTION PIPELINE')
    print('═'*64)
    print(f'  Input video  : {video_path}')
    print(f'  Output video : {output_video}')
    print(f'  Direction    : {args.dir.upper()}')
    print(f'  Session ID   : {session_id}')
    print('═'*64)

    # ── Init DB ───────────────────────────────────────────────
    os.makedirs('output', exist_ok=True)
    db.init_db()
    db.start_session(session_id, video_path, args.dir)

    # ── Load models ───────────────────────────────────────────
    print('\n[Step 1/5]  Loading models...')
    moto_det  = MotoDetector(args.moto_model,  conf=args.conf, iou=args.iou)
    plate_det = PlateDetector(args.plate_model, conf=0.25,
                              use_ocr=not args.no_ocr, gpu=False)

    # ── Open input video ──────────────────────────────────────
    print('\n[Step 2/5]  Opening video...')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  ERROR: Cannot read video: {video_path}')
        sys.exit(1)

    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_f / fps_src if fps_src > 0 else 0
    print(f'  Resolution  : {W} × {H}')
    print(f'  FPS         : {fps_src:.1f}')
    print(f'  Frames      : {total_f}')
    print(f'  Duration    : {duration:.1f} s  ({duration/60:.1f} min)')

    # ── Output video writer ───────────────────────────────────
    print('\n[Step 3/5]  Setting up output writer...')
    os.makedirs('output', exist_ok=True)

    # Try codecs in order until one works.
    # XVID writes to .avi and works on every Windows OpenCV build.
    codec_candidates = [
        ('mp4v', output_video),                          # .mp4 with mp4v
        ('XVID', output_video.replace('.mp4', '.avi')),  # .avi  ← always works
    ]

    writer     = None
    codec_name = None
    for fourcc_str, out_path in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w      = cv2.VideoWriter(out_path, fourcc, fps_src, (W, H))
        if w.isOpened():
            writer       = w
            codec_name   = fourcc_str
            output_video = out_path   # update so finish message is correct
            break
        w.release()

    if writer is None or not writer.isOpened():
        print('\n  ERROR: Could not open any video writer codec.')
        print('  Install opencv-python and make sure ffmpeg is available.')
        sys.exit(1)

    print(f'  Writing annotated video → {output_video}  [{codec_name}]')

    # ── Tracker + helper objects ──────────────────────────────
    print('\n[Step 4/5]  Setting up tracker and CSV writers...')
    byte_tracker = sv.ByteTrack(
        track_activation_threshold = args.conf,
        lost_track_buffer          = 40,
        minimum_matching_threshold = 0.80,
        frame_rate                 = int(fps_src),
    )
    dir_tracker = DirectionTracker(
        correct_dir = 1 if args.dir == 'right' else -1
    )
    roi_poly  = build_roi(W, H)
    plate_mem = PlateMemory()

    session_csv_dir = os.path.join('output', 'reports', session_id)
    csv_writer      = LiveCSVWriter(session_csv_dir)

    os.makedirs(os.path.join('output', 'snapshots'), exist_ok=True)
    recorder = ViolationRecorder(
        session_id   = session_id,
        video_source = video_path,
        fps          = fps_src,
        snapshot_dir = os.path.join('output', 'snapshots'),
        csv_writer   = csv_writer,
    )

    # ══════════════════════════════════════════════════════════
    # FRAME LOOP
    # ══════════════════════════════════════════════════════════
    print('\n[Step 5/5]  Processing frames...\n')
    fps_disp   = 0.0
    prev_t     = time.time()
    frame_no   = 0
    total_veh  = 0
    t_start    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break                      # end of video
        frame_no += 1

        # Frame skip (optional speedup)
        if frame_no % args.skip != 0:
            writer.write(frame)        # write original (no annotation) on skipped frames
            continue

        # ── Run moto model on full frame ───────────────────────
        moto_dets = moto_det.detect(frame)

        # ── Feed detections into ByteTrack ────────────────────
        if moto_dets:
            boxes   = np.array([[d['x1'], d['y1'], d['x2'], d['y2']]
                                 for d in moto_dets], dtype=float)
            confs   = np.array([d['conf'] for d in moto_dets], dtype=float)
            sv_dets = sv.Detections(
                xyxy       = boxes,
                confidence = confs,
                class_id   = np.zeros(len(moto_dets), dtype=int),
            )
            sv_dets = byte_tracker.update_with_detections(sv_dets)
        else:
            sv_dets = sv.Detections.empty()
            byte_tracker.update_with_detections(sv_dets)

        draw_roi(frame, roi_poly)

        active_ids = set()
        wrong_ids  = set()
        nohelm_ids = set()

        # ── Per-detection ──────────────────────────────────────
        for i, (box, _, conf_val, cls_id, tid, _) in enumerate(sv_dets):
            if tid is None:
                continue
            tid  = int(tid)
            x1, y1, x2, y2 = map(int, box)
            cx   = (x1 + x2) // 2
            cy   = (y1 + y2) // 2
            active_ids.add(tid)
            total_veh = max(total_veh, len(active_ids))

            # Align with moto_dets list by index
            moto = moto_dets[i] if i < len(moto_dets) else {}
            vehicle_class = moto.get('cls_name', 'Motorcycle')
            helmet_status = moto.get('helmet_status', 'Motorcycle')
            confidence    = moto.get('conf', float(conf_val))
            is_nohelm     = helmet_status == 'NO HELMET'
            bbox          = (x1, y1, x2, y2)

            if not in_roi(cx, cy, roi_poly):
                continue

            # ── Direction ──────────────────────────────────────
            direction = dir_tracker.update(tid, cx, cy)
            is_wrong  = direction == 'wrong'
            box_color = (C['wrong']   if is_wrong   else
                         C['correct'] if direction == 'correct' else
                         C['tracking'])

            draw_trail(frame, dir_tracker.histories[tid], box_color)

            # ── Vehicle label ──────────────────────────────────
            dir_tag = ('WRONG DIR' if is_wrong   else
                       'OK'        if direction == 'correct' else '...')
            label   = f'ID:{tid}  {vehicle_class}  [{dir_tag}]'
            draw_vehicle_box(frame, x1, y1, x2, y2, box_color, label,
                             thick=3 if (is_wrong or is_nohelm) else 2)

            # ── Helmet tag ─────────────────────────────────────
            tag_y = y2 + 2
            if helmet_status != 'Motorcycle':
                hcol = C['nohelm'] if is_nohelm else C['helmet']
                draw_tag(frame, x1, tag_y, helmet_status, hcol)
                tag_y += 24

            # ── License plate — always detect & draw ──────────
            plate_detected = False
            best_text      = plate_mem.get_text(tid)   # best OCR so far
            prev_box       = plate_mem.get_box(tid)

            plates = plate_det.detect(frame, x1, y1, x2, y2)

            if plates:
                # Use highest-confidence plate from model
                plates.sort(key=lambda p: p['conf'], reverse=True)
                best = plates[0]
                plate_detected = True

                improved, best_text = plate_mem.update(
                    tid, best['plate_text'],
                    best['px1'], best['py1'],
                    best['px2'], best['py2'],
                )
                if improved and best_text:
                    # Patch DB + CSV with newly read plate number
                    recorder.patch_plate(tid, best_text)

                # Draw plate box on video frame
                draw_plate(frame,
                           best['px1'], best['py1'],
                           best['px2'], best['py2'],
                           best_text)           # always show best known text

            elif best_text:
                # Plate model didn't fire this frame but we remember the text —
                # show it as a small tag so it's always visible on the video
                draw_tag(frame, x1, tag_y, f'LP: {best_text}', C['plate'])

            # ── Record violation ───────────────────────────────
            if is_wrong and is_nohelm:
                wrong_ids.add(tid)
                nohelm_ids.add(tid)
                recorder.record(frame, frame_no, tid, vehicle_class,
                                VT_BOTH, direction, helmet_status,
                                plate_detected, best_text, bbox, confidence)

            elif is_wrong:
                wrong_ids.add(tid)
                recorder.record(frame, frame_no, tid, vehicle_class,
                                VT_WRONG, direction, helmet_status,
                                plate_detected, best_text, bbox, confidence)

            elif is_nohelm:
                nohelm_ids.add(tid)
                recorder.record(frame, frame_no, tid, vehicle_class,
                                VT_HELM, direction, helmet_status,
                                plate_detected, best_text, bbox, confidence)

        # ── Cleanup stale tracks ───────────────────────────────
        dir_tracker.cleanup(active_ids)

        # ── HUD + alert banner ────────────────────────────────
        now      = time.time()
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t   = now

        draw_hud(frame, fps_disp,
                 len(wrong_ids), len(nohelm_ids),
                 len(sv_dets), args.dir,
                 frame_no, total_f, plate_mem.count_read())
        draw_alert_banner(frame, wrong_ids, nohelm_ids)

        # ── Write annotated frame to output video ─────────────
        writer.write(frame)

        # ── Optional preview window ────────────────────────────
        if args.preview:
            cv2.imshow(f'Processing: {video_name}   [Q = stop]', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\n  Preview stopped by user.')
                break

        # ── Progress print every 5 seconds ────────────────────
        elapsed = time.time() - t_start
        if frame_no % max(int(fps_src * 5), 1) == 0:
            pct  = frame_no / total_f * 100 if total_f > 0 else 0
            eta  = (elapsed / frame_no * (total_f - frame_no)
                    if frame_no > 0 and total_f > 0 else 0)
            print(f'  [{pct:5.1f}%]  frame {frame_no}/{total_f}  '
                  f'fps={fps_disp:4.1f}  '
                  f'wrong={len(wrong_ids)}  '
                  f'nohelm={len(nohelm_ids)}  '
                  f'plates={plate_mem.count_read()}  '
                  f'ETA={eta:.0f}s')

    # ══════════════════════════════════════════════════════════
    # FINISHED
    # ══════════════════════════════════════════════════════════
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    csv_writer.close()

    # Finalise session in DB
    summary = db.fetch_summary(session_id)
    total_v = sum(summary.values())
    db.end_session(session_id, frame_no, total_v)

    elapsed_total = time.time() - t_start

    print('\n' + '═'*64)
    print('  PROCESSING COMPLETE')
    print('═'*64)
    print(f'  Input video      : {video_path}')
    print(f'  Output video     : {output_video}')
    print(f'  Total frames     : {frame_no}')
    print(f'  Processing time  : {elapsed_total:.1f}s  ({elapsed_total/60:.1f} min)')
    print(f'  Avg speed        : {frame_no/elapsed_total:.1f} fps')
    print()
    print('  VIOLATIONS DETECTED:')
    print(f'  {"─"*36}')
    for vt, n in summary.items():
        label = {VT_WRONG: 'Wrong Direction',
                 VT_HELM:  'No Helmet',
                 VT_BOTH:  'Both (wrong dir + no helmet)'}.get(vt, vt)
        print(f'  {label:<30} : {n}')
    print(f'  {"─"*36}')
    print(f'  Total flags      : {total_v}')
    print(f'  Unique plates    : {plate_mem.count_read()}')
    print()
    print('  OUTPUT FILES:')
    print(f'  Annotated video  : {output_video}')
    print(f'  Database         : output/violations.db')
    print(f'  CSVs             : {session_csv_dir}/')
    print(f'    ├─ wrong_direction.csv')
    print(f'    ├─ no_helmet.csv')
    print(f'    ├─ both_flags.csv')
    print(f'    └─ all_violations.csv')
    print(f'  Snapshots        : output/snapshots/')
    print('═'*64)
    print()
    print(f'  Re-export CSVs anytime:')
    print(f'    python report.py --session {session_id}')
    print(f'    python report.py --list')
    print(f'    python report.py --plate <NUMBER>')
    print()


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    args = get_args()
    process_video(args)




