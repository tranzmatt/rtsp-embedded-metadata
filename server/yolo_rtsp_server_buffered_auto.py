#!/usr/bin/env python3
"""
YOLO RTSP Server — Frame-buffer synchronized version with adaptive drop control.
Keeps per-frame YOLO alignment while automatically adjusting buffer limits
to balance throughput and latency.
"""

import cv2, time, threading, json, struct, socket, subprocess, argparse, statistics
from datetime import datetime
from ultralytics import YOLO

try:
    import gpsd
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False


# -----------------------------------------------------
#  Video / Detection / Metadata
# -----------------------------------------------------
class VideoSource:
    def __init__(self, source):
        print(f"Opening: {source}")
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open: {source}")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame")
        self.width, self.height = frame.shape[1], frame.shape[0]
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        print(f"✓ Video: {self.width}x{self.height} @ {self.fps}fps")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class YOLODetector:
    def __init__(self, model_name, conf=0.5):
        print(f"Loading YOLO: {model_name}")
        self.model = YOLO(model_name)
        self.conf = conf
        self.results = {}
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        self.avg_times = []  # rolling detection times
        threading.Thread(target=self._worker, daemon=True).start()
        print("✓ YOLO loaded")

    def _worker(self):
        while self.running:
            if not self.queue:
                time.sleep(0.005)
                continue
            with self.lock:
                fid, frame, ts = self.queue.pop(0)
            t0 = time.time()
            res = self.model(frame, conf=self.conf, verbose=False)
            t_det = time.time() - t0
            self.avg_times.append(t_det)
            if len(self.avg_times) > 100:
                self.avg_times.pop(0)

            detections = []
            for r in res:
                for b in r.boxes:
                    detections.append({
                        "class": r.names[int(b.cls)],
                        "conf": round(float(b.conf), 2),
                        "bbox": [int(x) for x in b.xyxy[0]]
                    })
            self.results[fid] = {
                "detections": detections,
                "timestamp": ts,
                "detection_time": t_det
            }

    def queue_frame(self, fid, frame, ts):
        with self.lock:
            # if queue full, drop oldest
            if len(self.queue) >= 6:
                self.queue.pop(0)
            self.queue.append((fid, frame, ts))

    def pop_result(self, fid):
        return self.results.pop(fid, None)

    def avg_detection_time(self):
        return statistics.mean(self.avg_times) if self.avg_times else 0.1

    def stop(self):
        self.running = False


class GPSReader:
    def __init__(self):
        self.data = {}
        if not GPS_AVAILABLE:
            self.enabled = False
            return
        self.enabled = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            gpsd.connect()
        except Exception as e:
            print(f"⚠ GPS failed: {e}")
            self.enabled = False
            return
        while True:
            try:
                p = gpsd.get_current()
                self.data = {
                    "lat": getattr(p, "lat", None),
                    "lon": getattr(p, "lon", None),
                    "alt": getattr(p, "alt", None),
                    "speed": getattr(p, "hspeed", None)
                }
            except:
                time.sleep(1)

    def get(self):
        return self.data if self.enabled else {}


class RTSPServer:
    def __init__(self, width, height, fps, port=8554):
        self.start = time.time()
        cmd = [
            "ffmpeg", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast",
            "-tune", "zerolatency", "-pix_fmt", "yuv420p",
            "-g", str(fps * 2), "-f", "rtsp", "-rtsp_transport", "tcp",
            "-listen", "1", f"rtsp://0.0.0.0:{port}/live"
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.meta_port = port + 1000
        self.meta_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"✓ RTSP: rtsp://0.0.0.0:{port}/live  (metadata UDP {self.meta_port})")

    def timestamp(self): return time.time() - self.start

    def write_frame(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
            self.proc.stdin.flush()
        except BrokenPipeError:
            pass

    def send_metadata(self, meta):
        data = json.dumps(meta).encode()
        pkt = struct.pack("!I", len(data)) + data
        self.meta_sock.sendto(pkt, ("127.0.0.1", self.meta_port))

    def close(self):
        self.proc.terminate()
        self.meta_sock.close()


# -----------------------------------------------------
#  Main loop
# -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--disable-gps", action="store_true")
    args = ap.parse_args()

    video = VideoSource(args.input)
    yolo = YOLODetector(args.model, args.conf)
    gps = GPSReader() if not args.disable_gps else None
    server = RTSPServer(video.width, video.height, video.fps, args.port)

    frame_buffer = {}
    frame_count = 0

    print("\nServer running — buffering frames until YOLO results are ready.\n")

    try:
        while True:
            frame = video.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            ts = server.timestamp()

            # Queue and buffer
            yolo.queue_frame(frame_count, frame.copy(), ts)
            frame_buffer[frame_count] = (frame, ts)

            # Compute adaptive buffer size target
            avg_det = yolo.avg_detection_time()  # seconds
            # desired latency ≈ 2 × YOLO time or ~1 sec max
            max_buffer = int(min(video.fps * min(avg_det * 2, 1.0), 50))
            if max_buffer < 5:
                max_buffer = 5

            # Drop oldest if buffer exceeds target
            if len(frame_buffer) > max_buffer:
                oldest = min(frame_buffer)
                print(f"⚠ Dropping frame {oldest} (buffer {len(frame_buffer)} > {max_buffer})")
                frame_buffer.pop(oldest)

            # Process ready detections
            ready_ids = [fid for fid in list(frame_buffer.keys())
                         if fid in yolo.results]
            for fid in sorted(ready_ids):
                result = yolo.pop_result(fid)
                frame_send, pts = frame_buffer.pop(fid)
                metadata = {
                    "utc": datetime.utcnow().isoformat() + "Z",
                    "frame_id": fid,
                    "pts": pts,
                    "yolo": {
                        "analyzed_frame_id": fid,
                        "analyzed_pts": pts,
                        "detection_time": result["detection_time"],
                        "count": len(result["detections"]),
                        "detections": result["detections"]
                    }
                }

                # --- NEW: latency monitor ---
                latency = server.timestamp() - pts
                print(f"[Frame {fid}] latency {latency*1000:.0f} ms "
                      f"(avg_det {yolo.avg_detection_time()*1000:.0f} ms, "
                      f"buffer {len(frame_buffer)})")

                if gps:
                    gps_data = gps.get()
                    if gps_data:
                        metadata["gps"] = gps_data
                server.send_metadata(metadata)
                server.write_frame(frame_send)

            time.sleep(1.0 / video.fps)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        video.release()
        yolo.stop()
        server.close()
        print(f"✓ Done ({frame_count} frames)")


if __name__ == "__main__":
    main()

