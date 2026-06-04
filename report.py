# ╔══════════════════════════════════════════════════════════════════════╗
# ║  report.py  —  CSV Report Generator                                  ║
# ║  4 per-flag CSVs, each row contains plate_text when available        ║
# ║  By Musaawar Khan                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# STANDALONE USAGE:
#   python report.py                       ← latest session
#   python report.py --session SESSION_ID
#   python report.py --all                 ← every session combined
#   python report.py --plate ABC123        ← plate search
#   python report.py --list                ← list all sessions
# ══════════════════════════════════════════════════════════════════════

import csv
import os
import argparse
from datetime import datetime
from database import fetch_violations, fetch_summary, fetch_sessions, plate_lookup

REPORT_DIR = os.path.join('output', 'reports')

# ─────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS
# plate_text appears in every CSV so the operator always sees it
# ─────────────────────────────────────────────────────────────────────

COLS_MASTER = [
    'id', 'timestamp', 'session_id', 'video_source',
    'frame_number', 'frame_time_sec',
    'track_id', 'vehicle_class',
    'violation_type', 'direction', 'helmet_status',
    'plate_detected', 'plate_text', 'plate_updated_at',
    'snapshot_path', 'confidence',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'reviewed', 'notes',
]

COLS_WRONG_DIR = [
    'id', 'timestamp', 'frame_number', 'frame_time_sec',
    'track_id', 'vehicle_class',
    'direction',
    'plate_detected', 'plate_text',    # ← plate always here
    'snapshot_path', 'confidence',
]

COLS_NO_HELMET = [
    'id', 'timestamp', 'frame_number', 'frame_time_sec',
    'track_id', 'vehicle_class',
    'helmet_status',
    'plate_detected', 'plate_text',    # ← plate always here
    'snapshot_path', 'confidence',
]

COLS_BOTH = [
    'id', 'timestamp', 'frame_number', 'frame_time_sec',
    'track_id', 'vehicle_class',
    'direction', 'helmet_status',
    'plate_detected', 'plate_text',    # ← plate always here
    'snapshot_path', 'confidence',
]


# ─────────────────────────────────────────────────────────────────────
# INTERNAL CSV WRITE
# ─────────────────────────────────────────────────────────────────────

def _write_csv(rows: list, path: str, columns: list) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _ts():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ─────────────────────────────────────────────────────────────────────
# LIVE CSV WRITER
# Keeps four CSV files open and appends a row the instant a flag fires.
# plate_text in the row is whatever is known at that moment; a later
# OCR improvement is handled by pipeline.py calling csv_writer.update_plate().
# ─────────────────────────────────────────────────────────────────────

class LiveCSVWriter:
    """
    Opens all 4 CSVs at session start.
    .write(row, violation_type)   → append row immediately
    .update_plate(track_id, text) → re-write plate_text for all rows
                                     of that track_id (in-memory patch)
    .close()                      → flush + close all files
    """

    def __init__(self, session_dir: str):
        os.makedirs(session_dir, exist_ok=True)
        self._dir     = session_dir
        self._files   = {}
        self._writers = {}
        self._rows    = {}   # key → list of row dicts (for plate patching)

        specs = [
            ('WRONG_DIRECTION', 'wrong_direction.csv', COLS_WRONG_DIR),
            ('NO_HELMET',       'no_helmet.csv',        COLS_NO_HELMET),
            ('BOTH',            'both_flags.csv',       COLS_BOTH),
            ('ALL',             'all_violations.csv',   COLS_MASTER),
        ]
        for key, fname, cols in specs:
            path = os.path.join(session_dir, fname)
            fh   = open(path, 'w', newline='', encoding='utf-8')
            wr   = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            wr.writeheader()
            fh.flush()
            self._files[key]   = fh
            self._writers[key] = (wr, cols, path)
            self._rows[key]    = []
            print(f'  [CSV] Live → {path}')

    def write(self, row: dict, violation_type: str):
        """Append row to the matching type CSV and the master CSV."""
        for key in (violation_type.upper(), 'ALL'):
            if key in self._writers:
                wr, _, _ = self._writers[key]
                wr.writerow(row)
                self._files[key].flush()
                self._rows[key].append(row)   # keep in-memory copy

    def update_plate(self, track_id: int, plate_text: str):
        """
        When OCR finally reads a plate for track_id, retroactively patch
        all in-memory rows for that track, then rewrite the CSV file
        from scratch so the plate number appears in every relevant row.
        """
        pt = plate_text.upper().strip()
        if not pt:
            return

        patched_any = False
        for key, rows in self._rows.items():
            changed = False
            for row in rows:
                if row.get('track_id') == track_id:
                    existing = (row.get('plate_text') or '').strip()
                    # Only overwrite if new text is longer / current is empty
                    if not existing or len(pt) > len(existing):
                        row['plate_text']    = pt
                        row['plate_detected'] = 1
                        changed      = True
                        patched_any  = True

            if changed:
                # Rewrite entire CSV file with patched data
                wr, cols, path = self._writers[key]
                self._files[key].close()
                fh  = open(path, 'w', newline='', encoding='utf-8')
                new_wr = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
                new_wr.writeheader()
                new_wr.writerows(rows)
                fh.flush()
                self._files[key]   = fh
                self._writers[key] = (new_wr, cols, path)

        if patched_any:
            print(f'  [CSV] Plate "{pt}" patched into rows for ID:{track_id}')

    def close(self):
        for fh in self._files.values():
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
        print(f'  [CSV] Reports closed → {self._dir}/')


# ─────────────────────────────────────────────────────────────────────
# POST-RUN EXPORT FROM DB
# ─────────────────────────────────────────────────────────────────────

def export_reports(session_id: str = None, label: str = 'session') -> dict:
    """Pull records from DB and write all 4 CSVs + summary TXT."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    prefix  = os.path.join(REPORT_DIR, f'{_ts()}_{label}')
    all_rows = fetch_violations(session_id)
    wd_rows  = [r for r in all_rows if r['violation_type'] == 'WRONG_DIRECTION']
    nh_rows  = [r for r in all_rows if r['violation_type'] == 'NO_HELMET']
    bt_rows  = [r for r in all_rows if r['violation_type'] == 'BOTH']
    summary  = fetch_summary(session_id)
    results  = {}

    specs = [
        ('wrong_direction', f'{prefix}_wrong_direction.csv', wd_rows,  COLS_WRONG_DIR),
        ('no_helmet',       f'{prefix}_no_helmet.csv',        nh_rows,  COLS_NO_HELMET),
        ('both_flags',      f'{prefix}_both_flags.csv',       bt_rows,  COLS_BOTH),
        ('all_violations',  f'{prefix}_all_violations.csv',   all_rows, COLS_MASTER),
    ]
    for key, path, rows, cols in specs:
        n = _write_csv(rows, path, cols)
        results[key] = (path, n)
        print(f'  [CSV] {key.replace("_"," ").title():<22} → '
              f'{os.path.basename(path)}  ({n} rows)')

    txt = f'{prefix}_summary.txt'
    _write_summary(txt, summary, all_rows, session_id)
    results['summary'] = (txt, 1)
    print(f'  [TXT] Summary             → {os.path.basename(txt)}')
    return results


def _write_summary(path, summary, rows, session_id):
    from collections import Counter
    total   = len(rows)
    wd      = summary.get('WRONG_DIRECTION', 0)
    nh      = summary.get('NO_HELMET', 0)
    both    = summary.get('BOTH', 0)
    top_ids = Counter(r['track_id'] for r in rows).most_common(5)
    plates  = sorted({r['plate_text'] for r in rows if r.get('plate_text','').strip()})

    with open(path, 'w', encoding='utf-8') as f:
        f.write('=' * 58 + '\n')
        f.write('   VIOLATION DETECTION — SESSION SUMMARY\n')
        f.write('=' * 58 + '\n')
        f.write(f'  Generated  : {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        if session_id:
            f.write(f'  Session    : {session_id}\n')
        f.write('\n  VIOLATIONS BY TYPE\n  ' + '-'*36 + '\n')
        f.write(f'  Wrong Direction    : {wd:>5}\n')
        f.write(f'  No Helmet          : {nh:>5}\n')
        f.write(f'  Both               : {both:>5}\n')
        f.write(f'  ─────────────────────────\n')
        f.write(f'  TOTAL              : {total:>5}\n')
        f.write('\n  TOP REPEAT OFFENDERS\n  ' + '-'*36 + '\n')
        for tid, cnt in top_ids:
            f.write(f'  Track ID {tid:<6} → {cnt} flag(s)\n')
        f.write('\n  LICENSE PLATES FLAGGED\n  ' + '-'*36 + '\n')
        if plates:
            for p in plates[:30]:
                f.write(f'  {p}\n')
            if len(plates) > 30:
                f.write(f'  ... and {len(plates)-30} more\n')
        else:
            f.write('  (none read by OCR)\n')
        f.write('\n' + '=' * 58 + '\n')


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', default=None)
    p.add_argument('--all',     action='store_true')
    p.add_argument('--plate',   default=None)
    p.add_argument('--list',    action='store_true')
    args = p.parse_args()

    print('\n' + '='*58 + '\n  VIOLATION REPORT GENERATOR\n' + '='*58 + '\n')

    if args.list:
        for s in fetch_sessions():
            print(f'  {s["session_id"]:<30} {s["started_at"]:<22} '
                  f'flags={s["total_violations"]}')
        return

    if args.plate:
        rows = plate_lookup(args.plate)
        path = os.path.join(REPORT_DIR, f'{_ts()}_plate_{args.plate.upper()}.csv')
        n    = _write_csv(rows, path, COLS_MASTER)
        print(f'  Plate "{args.plate}" → {path}  ({n} rows)')
        return

    if args.all:
        export_reports(session_id=None, label='ALL')
    elif args.session:
        export_reports(session_id=args.session, label=args.session[:16])
    else:
        sessions = fetch_sessions()
        if not sessions:
            print('  No sessions found. Run pipeline.py first.')
            return
        sid = sessions[0]['session_id']
        print(f'  Latest session: {sid}')
        export_reports(session_id=sid, label=sid[:16])

    print('\n  Done.')


if __name__ == '__main__':
    main()
