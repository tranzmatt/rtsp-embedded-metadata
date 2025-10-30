# RTSP Embedded Metadata Streaming (YOLO + GStreamer)

This project demonstrates how to embed YOLOv8 detection metadata **into the same RTSP stream** as the video—no side UDP channel—using GStreamer dual tracks.  
The server runs YOLO, injects detections into the metadata track, and optionally pushes to a remote RTSP host.  
The client reads both tracks, prints the JSON metadata, and can optionally overlay detections.

---

## 🚀 Features
- YOLOv8 inference on live or file-based video input.
- Metadata embedded in-band in the RTSP stream (no UDP side channel).
- Supports **local hosting** or **remote push** (`rtspclientsink`).
- Automatic detection of encoder:
  - Jetson / DeepStream → `nvv4l2h264enc`
  - x86 / NVENC → `nvh264enc`
  - Fallback → `x264enc`
- Client prints detections as **pretty-printed JSON**.
- Optional live overlay via OpenCV (`--overlay`).

---

## 🧠 Architecture Overview
```
[YOLOv8] → [Video + JSON Metadata] → [GStreamer Dual Track RTSP]
                ↳ pay0: video/H264
                ↳ pay1: application/x-gst (JSON)
```

**Client:**
```
rtspsrc → (track 0 → video sink)
        → (track 1 → metadata appsink)
```

---

## 🖥️ Setup

### 1. Install dependencies
```bash
sudo apt install python3-gi gir1.2-gst-1.0 gstreamer1.0-tools   gstreamer1.0-plugins-base gstreamer1.0-plugins-good   gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly   gstreamer1.0-libav gstreamer1.0-rtsp

pip install -r requirements.txt
```

### 2. Verify NVIDIA encoder (Jetson / DeepStream)
```bash
gst-inspect-1.0 nvv4l2h264enc
```
If found, the pipeline will auto-select it.

---

## 🧩 Usage

### 🖥️ Start the Server

**Local RTSP Server**
```bash
python gst_rtsp_dualtrack_server_remote.py   --input http://64.191.148.57/mjpg/video.mjpg   --model yolov8n.pt   --port 8554   --path /live
```
→ Stream URL: `rtsp://localhost:8554/live`

**Push to Remote RTSP Host**
```bash
python gst_rtsp_dualtrack_server_remote.py   --input http://64.191.148.57/mjpg/video.mjpg   --model yolov8n.pt   --output rtsp://dev.imagery.comp-dev.org:8554/test-meta
```

---

### 🧭 Run the Client

**Print JSON metadata only**
```bash
python gst_rtsp_dualtrack_client_remote.py   --url rtsp://dev.imagery.comp-dev.org:8554/test-meta
```

**Show live video with detection overlay**
```bash
python gst_rtsp_dualtrack_client_remote.py   --url rtsp://dev.imagery.comp-dev.org:8554/test-meta   --overlay
```

**Headless mode (no video, just logs)**
```bash
python gst_rtsp_dualtrack_client_remote.py   --url rtsp://dev.imagery.comp-dev.org:8554/test-meta   --no-video
```

---

## 🧩 Example Output

**Console (JSON pretty print):**
```json
{
  "frame_id": 1024,
  "timestamp": 1730201522.819,
  "yolo": {
    "detections": [
      {"class": "person", "conf": 0.92, "bbox": [315, 57, 402, 291]},
      {"class": "dog", "conf": 0.87, "bbox": [78, 85, 225, 312]}
    ],
    "det_ms": 23.8
  }
}
```

**Overlay (when enabled):**
- Bounding boxes and labels drawn directly onto decoded frames.

---

## 🧰 Troubleshooting

| Problem | Fix |
|----------|------|
| “Failed to load plugin libgstrtspserver…” | `sudo apt install libgstrtspserver-1.0-0` |
| “No such element nvh264enc” | Use Jetson’s `nvv4l2h264enc` or install NVENC SDK |
| DeepStream warnings | Harmless if pipeline still runs |
| No detections printed | Ensure YOLO model file is valid and accessible |
| High latency | Lower `--latency` on client or `queue-size` on server |

---

## 🧩 License
MIT License © 2025 — Created for research & development of in-band RTSP metadata streaming.
