#!/usr/bin/env python3
"""
RTSP Client — Option 2 with overlay + latency HUD.

Keeps a short buffer of frames and aligns YOLO metadata by timestamp.
Draws detection boxes and shows Δ latency on the frame.
"""

import cv2, socket, struct, json, threading, time, argparse
from collections import deque


# -----------------------------------------------------
# Metadata receiver
# -----------------------------------------------------
class MetadataReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(1.0)
        self.latest = None
        threading.Thread(target=self._recv, daemon=True).start()
        print(f"✓ Listening for metadata on UDP {port}")

    def _recv(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                length = struct.unpack("!I", data[:4])[0]
                msg = json.loads(data[4:4 + length].decode())
                msg["_recv_time"] = time.time()
                self.latest = msg
            except socket.timeout:
                continue
            except Exception:
                pass

    def get(self):
        return self.latest


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--buffer-ms", type=int, default=800,
                    help="Client frame buffer window (ms)")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    rtsp_url = f"rtsp://{args.host}:{args.port}/live"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Failed to open RTSP stream")
        return

    meta = MetadataReceiver(args.port + 1000)
    frame_buf = deque()  # (timestamp, frame)
    buffer_window = args.buffer_ms / 1000.0
    last_print = time.time()

    print(f"✓ Connected to {rtsp_url}")
    print(f"Maintaining {args.buffer_ms} ms buffer for alignment\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.time()
            frame_buf.append((now, frame))
            # trim old frames
            while frame_buf and now - frame_buf[0][0] > buffer_window:
                frame_buf.popleft()

            m = meta.get()
            if not m or "yolo" not in m:
                if not args.no_display:
                    cv2.imshow("RTSP (Live)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            y = m["yolo"]
            target_pts = y.get("analyzed_pts")
            if target_pts is None or not frame_buf:
                continue

            # estimate relative lag between video clock and metadata
            # assumes pts values grow at real-time rate from server.start
            recv_now = m["_recv_time"]
            send_diff = m["pts"] - target_pts     # how far behind YOLO was
            est_capture_time = recv_now - send_diff

            # find closest frame by timestamp
            closest = min(frame_buf, key=lambda f: abs(f[0] - est_capture_time))
            disp = closest[1].copy()
            latency = (recv_now - target_pts) * 1000.0  # ms

            # draw detections
            for d in y.get("detections", []):
                x1, y1, x2, y2 = d["bbox"]
                label = f"{d['class']} {d['conf']:.2f}"
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # draw latency HUD (top-left corner)
            cv2.rectangle(disp, (5, 5), (170, 30), (0, 0, 0), -1)
            cv2.putText(disp, f"Δ {latency:5.0f} ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # periodic console log
            if time.time() - last_print > 1.0:
                print(f"Frame {m.get('frame_id')} : "
                      f"{y.get('count',0)} detections "
                      f"({y.get('detection_time',0)*1000:.0f} ms, "
                      f"Δ≈{latency:.0f} ms)")
                last_print = time.time()

            if not args.no_display:
                cv2.imshow("RTSP (Overlay + Latency)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

