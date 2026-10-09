# 🚦 Vehicle Violation Detection System

> **AI-powered real-time traffic violation monitoring — Wrong Direction Detection · Helmetless Rider Detection · License Plate Recognition · SMS & Email Alerts**

**By Musaawar Khan** | Final Year Project | UET Peshawar — Department of Data Science
Supervised by **Dr. Imran Khalil** | Group Partner: **Haseeb Aman (22PWDSC0060)**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Features](#-features)
4. [File Structure](#-file-structure)
5. [Requirements & Installation](#-requirements--installation)
6. [Model Training on Google Colab](#-model-training-on-google-colab)
7. [Running the Pipeline](#-running-the-pipeline)
8. [Desktop UI Guide](#-desktop-ui-guide)
9. [Alert System — SMS & Email](#-alert-system--sms--email)
10. [Database Schema](#-database-schema)
11. [CSV Reports](#-csv-reports)
12. [Output Files](#-output-files)
13. [Configuration Reference](#-configuration-reference)
14. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

This system detects traffic violations from video footage using custom-trained YOLOv8 models. It identifies three violation types in real time, reads license plate numbers via OCR, stores every flagged event in a SQLite database, exports per-rule CSV reports, and fires **SMS and email alerts** the moment a violation is detected.

| Violation | Description |
|---|---|
| `WRONG_DIRECTION` | Vehicle travelling against the legal traffic flow |
| `NO_HELMET` | Motorcycle rider without a helmet |
| `BOTH` | Rider going wrong direction AND without a helmet |

**Pipeline flow:**

```
Input Video
    │
    ▼
motorcycle_best.pt  ──→  Detect vehicles (helmet / helmetless / motorcycle)
    │
    ▼
ByteTrack           ──→  Assign persistent Track IDs across frames
    │
    ▼
DirectionTracker    ──→  Compute travel direction per vehicle (centroid history)
    │
    ▼
plate_best.pt       ──→  Detect license plate region inside vehicle crop
    │
    ▼
EasyOCR             ──→  Read plate number text
    │
    ├──→  Draw annotations on video frame
    ├──→  Write to SQLite database (violations.db)
    ├──→  Append to live CSV reports
    ├──→  Save snapshot JPEG
    └──→  🚨 Fire SMS + Email alert
```

---

## 🏗 System Architecture

```
project/
├── app.py                      ← Professional PyQt6 Desktop UI
├── pipeline.py                 ← Main detection pipeline (video → annotated output)
├── database.py                 ← SQLite wrapper (insert, query, update plate text)
├── report.py                   ← Live CSV writer + post-run export
├── alerts.py                   ← SMS (Twilio) + Email (SMTP) alert system
├── alert_config.py             ← Your credentials (SMS/email settings)
├── requirements.txt            ← All Python dependencies
│
├── models/
│   ├── motorcycle_best.pt      ← Your trained helmet/helmetless model
│   └── plate_best.pt           ← Your trained license plate model
│
├── train_motorcycle_colab.py   ← Colab training script for motorcycle model
├── train_plate_colab.py        ← Colab training script for plate model
│
└── output/                     ← Auto-created by pipeline
    ├── result_<video>.mp4      ← Annotated output video
    ├── violations.db           ← SQLite database
    ├── snapshots/              ← One JPEG per first-seen violation
    └── reports/<session_id>/   ← Per-session live CSVs
        ├── wrong_direction.csv
        ├── no_helmet.csv
        ├── both_flags.csv
        └── all_violations.csv
```

---

## ✨ Features

- **YOLOv8** object detection — your own trained weights, no COCO fallback
- **ByteTrack** multi-object tracking — stable IDs across frames
- **Centroid-history direction detection** — smooth, noise-resistant
- **License plate detection** — plate_best.pt crops the plate region inside every vehicle box
- **EasyOCR** — reads plate number text, automatically upscales tiny plates for better accuracy
- **Plate memory per track** — best OCR read persisted across all frames; patches DB and CSV retroactively when a better read arrives on a later frame
- **SQLite database** — every violation stored with full metadata including plate text, bbox, confidence, snapshot path
- **Live CSV** — four separate files written in real-time, one row per flag
- **PyQt6 Desktop UI** — Detection, Database Viewer, Reports, and Session History tabs
- **🚨 SMS alerts via Twilio** — text message sent to multiple numbers on every new violation
- **🚨 Email alerts via SMTP/Gmail** — dark-themed HTML email with snapshot attached on every flag
- **Progress bar with ETA** — shows processing speed and estimated time remaining
- **Lane ROI** — configurable region of interest polygon to focus on specific road lanes

---

## 📁 File Structure

| File | Purpose |
|---|---|
| `app.py` | PyQt6 desktop UI — run this to open the full application |
| `pipeline.py` | Core detection pipeline — processes video, writes all outputs |
| `database.py` | SQLite store — handles insert, update plate text, and all queries |
| `report.py` | LiveCSVWriter for real-time CSV output + post-run export tool |
| `alerts.py` | SMS (Twilio) + Email (SMTP/Gmail) alert dispatcher |
| `alert_config.py` | Your API keys and credentials — never commit this to Git |
| `train_motorcycle_colab.py` | YOLOv8 training on Colab for the helmet/helmetless model |
| `train_plate_colab.py` | YOLOv8 training on Colab for the license plate model |
| `requirements.txt` | All pip dependencies |

---

## ⚙ Requirements & Installation

### Python Version

Python **3.9 or higher** is required. Tested on Python 3.10 and 3.11.

### Step 1 — Install PyTorch (choose based on your hardware)

**CPU only (no GPU):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU — CUDA 11.8:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**NVIDIA GPU — CUDA 12.1:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Install all other dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Install alert dependencies

```bash
pip install twilio
```

Email alerts use Python's built-in `smtplib` — no extra package needed.

### Step 4 — Install the Desktop UI dependency

```bash
pip install PyQt6
```

### Linux only — if OpenCV fails to import

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Complete package reference

| Package | Min Version | Used For |
|---|---|---|
| `torch` | 2.0.0 | Running YOLO model inference |
| `torchvision` | 0.15.0 | Model image transforms |
| `ultralytics` | 8.0.0 | YOLOv8 — loads your `.pt` weight files |
| `supervision` | 0.18.0 | ByteTrack — assigns persistent Track IDs |
| `opencv-python` | 4.8.0 | Video reading, annotation drawing, output writing |
| `easyocr` | 1.7.0 | License plate OCR |
| `numpy` | 1.24.0 | Array maths and bounding box calculations |
| `PyQt6` | 6.5.0 | Desktop UI |
| `twilio` | 8.0.0 | SMS alerts |
| `sqlite3` | built-in | Violation database — no install needed |

---

## 🤖 Model Training on Google Colab

Both custom models must be trained on Colab before running the pipeline. Colab provides a free T4 GPU which is sufficient for training.

### Train the Motorcycle / Helmet Model

1. Open **Google Colab** → Runtime → Change runtime type → **T4 GPU**
2. Copy the contents of `train_motorcycle_colab.py` into Colab cells (each section marked `# CELL N` goes into its own cell)
3. Run all cells from top to bottom
4. After training completes, Cell 10 automatically downloads `best.pt` to your computer
5. Rename the downloaded file to `motorcycle_best.pt` and place it in the `models/` folder

What the training script does automatically:
- Downloads the motorcycle/helmet dataset from Roboflow
- Trains YOLOv8m for 80 epochs with augmentation
- Saves checkpoints to Google Drive so training auto-resumes if Colab disconnects
- Reports mAP, Precision, Recall, and F1 score after training

### Train the License Plate Model

1. Follow the same steps using `train_plate_colab.py`
2. After training, rename the downloaded `best.pt` to `plate_best.pt` and place it in `models/`

### Expected class names in your motorcycle model

The pipeline matches class names by partial string. Your model classes should include:

| Class name examples | Detected as |
|---|---|
| `helmet`, `with_helmet`, `has_helmet` | ✅ Helmet worn |
| `no_helmet`, `helmetless`, `without_helmet` | ❌ NO HELMET flag |
| `motorcycle`, `bike`, `rider` | Generic motorcycle — direction still tracked |

If your class names differ, edit the `_is_helmet()` and `_is_nohelmet()` methods inside `MotoDetector` in `pipeline.py`.

---

## ▶ Running the Pipeline

### Option A — Desktop UI (recommended)

```bash
python app.py
```

The UI opens with four tabs: Detection, Database, Reports, and Sessions. No command-line knowledge needed.

### Option B — Command Line

**Basic run — vehicles travel left to right:**

```bash
python pipeline.py --video traffic.mp4
```

**Road where vehicles travel right to left:**

```bash
python pipeline.py --video traffic.mp4 --dir left
```

**Show a live preview window while processing:**

```bash
python pipeline.py --video traffic.mp4 --preview
```

**Skip OCR for faster processing (no plate text):**

```bash
python pipeline.py --video traffic.mp4 --no-ocr
```

**Process every 2nd frame for speed on long videos:**

```bash
python pipeline.py --video traffic.mp4 --skip 2
```

**Full example with all options:**

```bash
python pipeline.py \
  --video       traffic.mp4 \
  --dir         left \
  --conf        0.35 \
  --skip        1 \
  --moto-model  models/motorcycle_best.pt \
  --plate-model models/plate_best.pt \
  --preview
```

### Export CSV reports after processing

```bash
python report.py                          # export latest session
python report.py --session SESSION_ID     # export a specific session
python report.py --all                    # export all sessions combined
python report.py --plate ABC123           # search by plate number
python report.py --list                   # list all sessions in DB
```

---

## 🖥 Desktop UI Guide

### Detection Tab

| Control | Description |
|---|---|
| Browse Video File | Select an `.mp4`, `.avi`, `.mov`, or `.mkv` file |
| Motorcycle model | Path to `motorcycle_best.pt` |
| Plate model | Path to `plate_best.pt` |
| Traffic direction | `right` = vehicles flow left→right, `left` = vehicles flow right→left |
| Confidence | YOLO detection threshold (0.10–0.90, default 0.35) |
| Frame skip | `1` = every frame, `2` = every other frame (faster) |
| Enable OCR | Uncheck to skip plate reading for faster processing |
| RUN PIPELINE | Starts processing — live log and stat cards update below |
| STOP | Terminates processing at the current frame |

Live stat cards update as processing runs:

- **Wrong Direction** — total vehicles flagged for wrong direction so far
- **No Helmet** — total helmetless riders flagged so far
- **Both Flags** — riders who have both violations
- **Plates Read** — unique license plates successfully read by OCR

### Database Tab

- Filter all records by violation type (Wrong Direction, No Helmet, Both, All)
- Filter by specific session using the session dropdown
- Search by license plate number — partial match works (e.g. searching `ABC` finds `ABC-1234`)
- All plate numbers displayed in cyan for easy scanning
- Click **Refresh** after a pipeline run to see new records

### Reports Tab

Click **Export** next to any CSV type to save it to a folder of your choice:

| CSV File | Contents |
|---|---|
| Wrong Direction | Track ID, vehicle class, direction, plate number, snapshot path, confidence |
| No Helmet | Track ID, vehicle class, helmet status, plate number, snapshot path, confidence |
| Both Flags | All fields for riders with both violations simultaneously |
| Master (All) | Every violation from every rule — all columns in one file |

### Sessions Tab

Shows a history of every pipeline run with start time, end time, total frames processed, video source, and total violation count.

---

## 🚨 Alert System — SMS & Email

The alert system fires automatically whenever a new violation is first detected. It runs in a background thread so it never slows down the video processing.

### How it works

1. A violation is detected for a vehicle (Track ID, violation type)
2. `fire_alert()` is called from inside the pipeline
3. A background thread starts — pipeline continues without waiting
4. The thread sends an SMS via Twilio to all configured phone numbers
5. The thread sends an HTML email via SMTP/Gmail to all configured email addresses
6. The violation snapshot JPEG is attached to the email
7. A per-vehicle cooldown prevents repeat alerts for the same vehicle

### Setup — Step 1: Get Twilio credentials (for SMS)

1. Sign up at [twilio.com](https://www.twilio.com) — free trial includes $15 credit
2. Go to the Twilio Console → Account Info
3. Copy your **Account SID** and **Auth Token**
4. Buy a Twilio phone number (or use the trial number)
5. On trial accounts, verify each recipient phone number before sending

### Setup — Step 2: Get Gmail App Password (for email)

1. Go to your Google Account → Security
2. Enable **2-Step Verification** if not already on
3. Go to Security → **App Passwords**
4. Generate a new App Password for "Mail"
5. Copy the 16-character password — this goes into `alert_config.py` as `SMTP_PASSWORD`

> Use the App Password, **not** your regular Gmail login password. Regular passwords will be rejected by Gmail's SMTP server.

### Setup — Step 3: Fill in `alert_config.py`

Open `alert_config.py` in your project root and fill in your credentials. The file is already created with placeholder values — replace each one:

| Field | What to put |
|---|---|
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID (starts with `AC`) |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token |
| `TWILIO_FROM_NUMBER` | Your Twilio phone number in E.164 format (e.g. `+12345678900`) |
| `ALERT_TO_NUMBERS` | List of recipient phone numbers in E.164 format |
| `SMTP_HOST` | `smtp.gmail.com` for Gmail |
| `SMTP_PORT` | `587` for TLS (use `465` for SSL if 587 is blocked) |
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASSWORD` | The 16-character Gmail App Password |
| `ALERT_TO_EMAILS` | List of recipient email addresses |
| `ALERT_COOLDOWN_SEC` | Seconds between repeat alerts for the same vehicle (default: `30`) |
| `ENABLE_SMS` | `True` to enable, `False` to disable |
| `ENABLE_EMAIL` | `True` to enable, `False` to disable |
| `ATTACH_SNAPSHOT` | `True` to attach the violation JPEG to emails |

> **Security:** Never commit `alert_config.py` to Git. Add it to your `.gitignore` file.

### Setup — Step 4: Test your credentials

Run the test without processing any video:

```bash
python alerts.py
```

This sends one test SMS and one test email using dummy data and prints whether each succeeded or failed. Check your phone and inbox to confirm delivery.

### Alert system summary

| Feature | Detail |
|---|---|
| SMS provider | Twilio — free trial gives $15 credit, enough for hundreds of messages |
| Email provider | Any SMTP server — Gmail recommended |
| Per-violation | One SMS + one email per new (Track ID × violation type) combination |
| Cooldown | Configurable seconds between repeat alerts for the same vehicle |
| Background thread | Alerts run in a daemon thread — never block the video pipeline |
| Snapshot attached | Violation frame JPEG is attached to every email automatically |
| Multiple recipients | Add as many phone numbers and email addresses as needed |
| Disable individually | Set `ENABLE_SMS = False` or `ENABLE_EMAIL = False` in `alert_config.py` |

### SMS message format

```
🚨 TRAFFIC VIOLATION DETECTED
Rule     : WRONG DIRECTION + NO HELMET
Track ID : 14
Plate    : ABC-1234
Frame    : 842
Video    : traffic_cam_01.mp4
Time     : 2024-06-15 14:32:01
```

### Email format

The email uses a dark-themed HTML layout showing:

- Large coloured header — red for wrong direction, amber for no helmet, orange for both
- Violation type displayed as a coloured badge
- Track ID highlighted in the violation colour
- License plate number in large cyan monospace font
- Frame number and video source filename
- Violation snapshot JPEG attached

---

## 🗄 Database Schema

### `violations` table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment row ID |
| `session_id` | TEXT | Unique ID for the pipeline run that created this row |
| `timestamp` | TEXT | ISO-8601 datetime when the violation was detected |
| `video_source` | TEXT | Input video file path |
| `frame_number` | INTEGER | Frame number where the violation was first detected |
| `frame_time_sec` | REAL | Time in video = frame number ÷ FPS |
| `track_id` | INTEGER | ByteTrack persistent vehicle ID |
| `vehicle_class` | TEXT | Class name from your model (e.g. `helmetless`, `motorcycle`) |
| `violation_type` | TEXT | `WRONG_DIRECTION`, `NO_HELMET`, or `BOTH` |
| `direction` | TEXT | `wrong`, `correct`, or `unknown` |
| `helmet_status` | TEXT | `Helmet`, `NO HELMET`, or `N/A` |
| `plate_detected` | INTEGER | `1` if the plate model found a plate box, `0` if not |
| `plate_text` | TEXT | OCR plate number — updated retroactively when OCR improves |
| `plate_updated_at` | TEXT | Timestamp of the last plate text update |
| `snapshot_path` | TEXT | Relative path to the saved JPEG snapshot |
| `confidence` | REAL | YOLO detection confidence score (0.0–1.0) |
| `bbox_x1` | INTEGER | Vehicle bounding box left edge |
| `bbox_y1` | INTEGER | Vehicle bounding box top edge |
| `bbox_x2` | INTEGER | Vehicle bounding box right edge |
| `bbox_y2` | INTEGER | Vehicle bounding box bottom edge |
| `reviewed` | INTEGER | `0` = pending review, `1` = reviewed by operator |
| `notes` | TEXT | Operator notes (editable) |

### `sessions` table

| Column | Type | Description |
|---|---|---|
| `session_id` | TEXT PK | Format: `YYYYMMDD_HHMMSS_xxxxxx` |
| `started_at` | TEXT | Session start timestamp |
| `ended_at` | TEXT | Session end timestamp |
| `video_source` | TEXT | Input video file path |
| `correct_direction` | TEXT | `right` or `left` — the legal traffic direction used |
| `total_frames` | INTEGER | Total frames processed in this session |
| `total_violations` | INTEGER | Total violation flags recorded in this session |

---

## 📊 CSV Reports

Four CSV files are generated per session. They are written live — a new row is appended the instant a violation is detected, not at the end of the video.

### `wrong_direction.csv`

Columns: `id`, `timestamp`, `frame_number`, `frame_time_sec`, `track_id`, `vehicle_class`, `direction`, `plate_detected`, `plate_text`, `snapshot_path`, `confidence`

### `no_helmet.csv`

Columns: `id`, `timestamp`, `frame_number`, `frame_time_sec`, `track_id`, `vehicle_class`, `helmet_status`, `plate_detected`, `plate_text`, `snapshot_path`, `confidence`

### `both_flags.csv`

Columns: `id`, `timestamp`, `frame_number`, `frame_time_sec`, `track_id`, `vehicle_class`, `direction`, `helmet_status`, `plate_detected`, `plate_text`, `snapshot_path`, `confidence`

### `all_violations.csv`

All columns from the database — every violation type in one master file. Best for importing into Excel or Power BI.

---

## 📂 Output Files

After a pipeline run, the `output/` folder contains:

```
output/
├── result_<videoname>.mp4          ← Annotated video with all overlays burned in
├── violations.db                   ← SQLite database (all sessions, all violations)
├── snapshots/
│   ├── wrong_direction_id5_f234.jpg
│   ├── no_helmet_id8_f412.jpg
│   └── both_id12_f890.jpg
└── reports/
    └── 20240615_143201_abc123/     ← One folder per session
        ├── wrong_direction.csv
        ├── no_helmet.csv
        ├── both_flags.csv
        └── all_violations.csv
```

### Annotated video overlays

- **Green bounding box** — vehicle travelling in the correct direction
- **Red bounding box** — wrong direction vehicle or helmetless rider
- **Cyan plate box** — license plate region detected by plate_best.pt, with OCR number shown above it in a dark badge
- **Green tag** — helmet confirmed below the vehicle box
- **Red tag** — NO HELMET below the vehicle box
- **Motion trail arrow** — shows the vehicle's movement path across recent frames
- **HUD panel (top-left)** — live FPS, traffic direction, vehicle count, violation counts, frame number, progress bar
- **Alert banner (bottom)** — red banner listing the Track IDs of currently active violators

---

## 🔧 Configuration Reference

### `pipeline.py` command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Input video file path |
| `--dir` | `right` | Legal traffic direction — `right` (left→right) or `left` (right→left) |
| `--conf` | `0.35` | YOLO confidence threshold (0.10 to 0.90) |
| `--iou` | `0.45` | NMS IoU threshold |
| `--skip` | `1` | Process every Nth frame — use `2` or `3` for faster processing |
| `--no-ocr` | off | Disable EasyOCR plate reading (faster) |
| `--preview` | off | Show a live OpenCV preview window while processing |
| `--moto-model` | `models/motorcycle_best.pt` | Path to the motorcycle/helmet model |
| `--plate-model` | `models/plate_best.pt` | Path to the license plate model |

### `alert_config.py` settings reference

| Setting | Type | Description |
|---|---|---|
| `TWILIO_ACCOUNT_SID` | string | Twilio Account SID from the console |
| `TWILIO_AUTH_TOKEN` | string | Twilio Auth Token |
| `TWILIO_FROM_NUMBER` | string | Your Twilio phone number in E.164 format |
| `ALERT_TO_NUMBERS` | list | Recipient phone numbers in E.164 format |
| `SMTP_HOST` | string | SMTP server hostname (`smtp.gmail.com`) |
| `SMTP_PORT` | integer | SMTP port — `587` for TLS, `465` for SSL |
| `SMTP_USER` | string | Sender email address |
| `SMTP_PASSWORD` | string | Gmail App Password (16 characters) |
| `ALERT_TO_EMAILS` | list | Recipient email addresses |
| `ALERT_COOLDOWN_SEC` | integer | Minimum seconds between repeat alerts for the same vehicle |
| `ENABLE_SMS` | boolean | `True` to send SMS, `False` to disable |
| `ENABLE_EMAIL` | boolean | `True` to send email, `False` to disable |
| `ATTACH_SNAPSHOT` | boolean | `True` to attach the JPEG snapshot to emails |

---

## 🛠 Troubleshooting

### Model not found

```
ERROR: motorcycle model not found: models/motorcycle_best.pt
```

Train the model on Colab using `train_motorcycle_colab.py`, download `best.pt`, rename it to `motorcycle_best.pt`, and place it in the `models/` folder. Same process for `plate_best.pt`.

### EasyOCR is slow on first run

This is normal. EasyOCR downloads approximately 200 MB of language model files on the first run. All subsequent runs use the cached files and start instantly.

### OpenCV import error on Linux

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### SMS not sending

- Verify the `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` values are correct
- On Twilio trial accounts, recipient numbers must be verified individually in the Twilio Console before messages can be sent to them
- Check your Twilio account balance — trial credit is $15
- Confirm `twilio` is installed by running `pip install twilio`
- Run `python alerts.py` to test your credentials in isolation

### Email not sending

- Use a Gmail **App Password** — not your regular Gmail password. Regular passwords are blocked by Gmail's SMTP
- Enable 2-Step Verification on the Google account first, then generate the App Password
- Port `587` (TLS) is the standard. If it is blocked by a corporate firewall, try port `465` with SSL instead
- Run `python alerts.py` to test email delivery independently of the pipeline

### Low FPS during video processing

- Use `--skip 2` or `--skip 3` to process fewer frames per second
- Use `--no-ocr` to disable EasyOCR — this is the slowest component on CPU
- If you have an NVIDIA GPU, set `gpu=True` in `MotoDetector` and `PlateDetector` inside `pipeline.py`
- Use the `yolov8n.pt` nano backbone when training for faster inference at lower accuracy

### License plate text is empty in the CSV

- OCR reads the plate on later frames and automatically patches older CSV and database rows when a better reading arrives — the CSV is not frozen at first-detection values
- Confirm the plate model is detecting boxes by enabling `--preview` and watching the cyan plate boxes
- Plates smaller than approximately 30×8 pixels in the frame are too small for reliable OCR — the video resolution or camera angle may need adjustment
- Lower `--conf` slightly (e.g. `0.25`) if the plate model is missing plates due to a high confidence threshold

### Direction detection flagging stationary vehicles

- Vehicles need to move at least 18 pixels horizontally across 8 frames before a direction is judged
- Stationary vehicles will stay in `unknown` state and will not be flagged
- If vehicles are being flagged too quickly, increase `min_frames` in the `DirectionTracker` class inside `pipeline.py`

### Wrong direction set incorrectly

- `--dir right` means vehicles **should** travel left to right. Vehicles going right to left will be flagged.
- `--dir left` means vehicles **should** travel right to left. Vehicles going left to right will be flagged.
- Check which direction is shown in the HUD panel on the annotated video

---

## 📄 License

This project is developed for academic purposes as a Final Year Project at **UET Peshawar, Department of Data Science**.

---

*Vehicle Violation Detection System · By Musaawar Khan · UET Peshawar*