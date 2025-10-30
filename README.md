# YOLO RTSP Stream with Embedded Metadata

This project demonstrates two complementary methods for embedding YOLO object-detection metadata into a live RTSP video stream and synchronizing it on the client side.

## Overview

| Mode | Description | Latency | Load |
|------|--------------|----------|------|
| Option 1 – Server Buffering | The server waits for YOLO detections before sending each frame. Perfect synchronization. | Higher (≈ YOLO inference time) | Heavy on server |
| Option 2 – Client Alignment | The server streams video immediately; YOLO runs asynchronously and sends metadata separately. The client buffers and aligns frames. | Low (sub-second) | Heavier on client |

Both modes share the same basic structure:

Camera / File → YOLO → Metadata (UDP)
                     ↘
                      RTSP (FFmpeg) → Client

## Requirements

* Python 3.8 or later
* ffmpeg (must be installed and available on PATH)
* Python packages:
  * ultralytics
  * opencv-python
  * gpsd-py3 (optional, for GPS data)

Install dependencies:

pip install ultralytics opencv-python gpsd-py3

## File Summary

| File | Purpose |
|------|----------|
| yolo_rtsp_server_buffered_auto.py | Option 1 (Server Buffering). Frames are delayed until YOLO results are ready. Adaptive buffer keeps latency stable. |
| yolo_rtsp_client_buffered_auto.py | Client for Option 1. Displays synchronized stream. |
| yolo_rtsp_server_streaming.py | Option 2 (Server Streaming). Streams immediately, sends YOLO metadata asynchronously. |
| yolo_rtsp_client_overlay_latency.py | Client for Option 2. Buffers frames, overlays YOLO detections, and shows real-time latency. |

## Usage

### Option 1 — Server-Side Buffering (Synchronous)

Provides perfect alignment between frames and detections, at the cost of higher latency.

# Terminal 1 – start the server
python yolo_rtsp_server_buffered_auto.py \
    --input http://64.191.148.57/mjpg/video.mjpg \
    --model yolov8n \
    --port 8554

# Terminal 2 – start the client
python yolo_rtsp_client_buffered_auto.py --port 8554

Notes:

* The default RTSP endpoint is rtsp://0.0.0.0:8554/live.
* Metadata is broadcast via UDP on port 9554 (8554 + 1000).
* The server dynamically adjusts its buffer size based on YOLO inference time.
* Console output shows per-frame latency and detections.

### Option 2 — Client-Side Alignment (Asynchronous)

Provides low-latency video playback with a small client buffer (default 700 ms) used to align metadata.

# Terminal 1 – start the server
python yolo_rtsp_server_streaming.py \
    --input http://64.191.148.57/mjpg/video.mjpg \
    --model yolov8n \
    --port 8554

# Terminal 2 – start the client
python yolo_rtsp_client_overlay_latency.py --port 8554 --buffer-ms 700

Notes:

* The server streams video immediately and runs YOLO in a background thread.
* Metadata is timestamped and sent independently via UDP.
* The client maintains a rolling frame buffer, finds the closest timestamp, draws bounding boxes, and displays a latency indicator on screen.
* Press q in the window to quit.

## Comparison

| Feature | Option 1 | Option 2 |
|----------|-----------|-----------|
| Latency | ≈ YOLO time (300–800 ms typical) | < 200 ms visual |
| Synchronization | Perfect (1 : 1 frame) | Near-perfect (± 1 frame) |
| Server Load | High (buffering + inference) | Low (inference only) |
| Client Load | Low | Moderate (buffer + overlay) |
| Use Case | Analytics, recording, dataset labeling | Live monitoring, dashboards |
| Scalability | Single-client focus | Many clients supported |

## Metadata Format

Each UDP metadata packet contains a JSON object:

{
  "utc": "2025-10-30T14:10:00.532Z",
  "frame_id": 312,
  "pts": 42.317,
  "yolo": {
    "analyzed_frame_id": 312,
    "analyzed_pts": 42.317,
    "detection_time": 0.091,
    "count": 3,
    "detections": [
      { "class": "person", "conf": 0.87, "bbox": [210, 100, 360, 480] },
      { "class": "dog", "conf": 0.72, "bbox": [420, 200, 520, 460] }
    ]
  },
  "gps": { "lat": 38.89, "lon": -77.03 }
}

## Tips

* Use --model yolov8n.pt or --model yolov8s.pt to select a YOLO model.
* Use --input 0 for a live webcam.
* In Option 1, latency equals YOLO inference time; choose a faster model to reduce lag.
* In Option 2, if detections appear slightly late, increase --buffer-ms.

## Future Enhancements

* Hybrid mode (server embeds metadata once ready; clients choose live or synchronized mode)
* WebSocket metadata transport
* Rolling latency graph on the client for debugging

