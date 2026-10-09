# Pipeline patch — enable dashboard live video feed

Add **one line** to `pipeline.py` wherever you write the annotated frame to the
OpenCV window (or just before `cv2.imshow`).  Find the section that looks like:

```python
# somewhere in your frame loop, after annotations are drawn on `frame`:
annotated = results[0].plot()   # or however you get your annotated frame

# ── ADD THIS LINE ──────────────────────────────────────────────────────
cv2.imwrite('output/latest_frame.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
# ───────────────────────────────────────────────────────────────────────

if show_preview:
    cv2.imshow('ViolationIQ', annotated)
```

`server.py` polls `output/latest_frame.jpg` every 50 ms and pushes new frames
to any connected browser tab via the MJPEG stream at `/api/video_feed`.

No other changes needed — `server.py` handles everything else automatically.
