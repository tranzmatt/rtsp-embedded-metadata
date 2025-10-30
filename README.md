# RTSP Embedded Metadata (YOLO + GPS + GStreamer)

This project demonstrates how to **embed YOLOv8 detections and optional GPSD telemetry directly into an RTSP stream** using a dual-track GStreamer pipeline.  
The result is a single stream containing both H.264 video and synchronized JSON metadata — no side channels or separate UDP sockets.

---

## 🚀 Features

- Real-time **YOLOv8 object detection**
- Metadata embedding directly into RTSP stream (same frames)
- Optional **GPSD integration** for geolocation data
- Local or **remote RTSP push**
- Smart encoder selection (DeepStream hardware → software fallback)
- Client supports **overlay rendering** or **pretty-printed JSON**
- Robust and reconnecting GPS polling (30s interval)

---

## 🧩 Files

| File | Description |
|------|--------------|
| `gst_rtsp_dualtrack_server_remote.py` | Unified YOLO + GPS RTSP server (push or local stream) |
| `gst_rtsp_dualtrack_client_remote.py` | RTSP client that prints or overlays detections + GPS |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

---

## 🧭 Server Usage

### Example: YOLO + GPSD RTSP Server (remote push)
```bash
python gst_rtsp_dualtrack_server_remote.py   --input http://64.191.148.57/mjpg/video.mjpg   --model yolov8n.pt   --output rtsp://dev.imagery.comp-dev.org:8554/test-meta   --gpsd --gps-host gps.example.net --gps-port 2947   --print-detections
```

### Example: Local RTSP Server (no push)
```bash
python gst_rtsp_dualtrack_server_remote.py   --input rtsp://camera/stream   --model yolov8n.pt   --port 8554 --path /live   --gpsd
```

Each frame’s metadata includes YOLO detections and GPS fix:

```json
{
  "frame_id": 145,
  "timestamp": 1730201801.84,
  "yolo": { "detections": [...], "det_ms": 27.3 },
  "gps": { "lat": 37.7749, "lon": -122.4194, "alt": 16.4, "speed": 0.1 }
}
[GPS] lat=37.7749 lon=-122.4194 alt=16.4 speed=0.1
```

---

## 🧠 Client Usage

The client receives the RTSP stream, extracts embedded JSON metadata, and either prints it prettily or overlays bounding boxes.

```bash
python gst_rtsp_dualtrack_client_remote.py   --url rtsp://dev.imagery.comp-dev.org:8554/test-meta   --overlay  # optional
```

---

## 🧰 Dependencies

Install dependencies:

```bash
sudo apt install python3-gi gir1.2-gst-1.0 gstreamer1.0-tools   gstreamer1.0-plugins-base gstreamer1.0-plugins-good   gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly   gstreamer1.0-libav gstreamer1.0-rtsp

pip install -r requirements.txt
```

> 💡 The `gps` package is only required if using `--gpsd`.

---

## ⚙️ Encoder Notes

The pipeline automatically chooses the best available encoder:
1. **nvv4l2h264enc** – NVIDIA DeepStream hardware encoder (Jetson, A100, etc.)
2. **nvh264enc** – NVENC hardware encoder (desktop GPUs)
3. **x264enc** – Software fallback (CPU-only systems)

---

## 🧾 Changelog

### 2025-10 — Unified YOLO + GPS Version
- Added GPSD integration with reconnect and top-level metadata
- Optional `--gpsd`, `--gps-host`, and `--gps-port` flags
- `[GPS] lat lon alt speed` printed alongside detections
- Replaced previous non-GPS and buffered variants
- Maintains backward compatibility for all prior CLI options
- Improved RTSP push and encoder selection logic

### 2025-09 — Buffered/Alignment Versions
- Frame synchronization between YOLO detections and video
- Reduced latency via frame queueing and timestamp matching

### 2025-08 — Initial Dual-Track Architecture
- Introduced GStreamer dual-track (H.264 + JSON metadata)
- Added client with overlay toggle and pretty JSON output

---

## 🧡 Credits

Developed for NVIDIA DeepStream + YOLOv8 workflows, designed to run on both Jetson and x86 systems.
