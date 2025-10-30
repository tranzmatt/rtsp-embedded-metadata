#!/usr/bin/env python3
"""
RTSP Client — For buffered server with drop policy.
Displays frames and synchronized YOLO metadata.
"""

import cv2, socket, struct, json, threading, time, argparse


class MetadataReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(1.0)
        self.meta = {}
        threading.Thread(target=self._recv, daemon=True).start()
        print(f"✓ Listening for metadata on UDP {port}")

    def _recv(self):
        while True:
            try:
                d, _ = self.sock.recvfrom(65535)
                l = struct.unpack("!I", d[:4])[0]
                js = json.loads(d[4:4 + l].decode())
                self.meta = js
            except socket.timeout:
                continue
            except Exception:
                pass

    def get(self):
        return self.meta.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    rtsp_url = f"rtsp://{args.host}:{args.port}/live"
    meta = MetadataReceiver(args.port + 1000)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Failed to open RTSP stream")
        return

    print(f"Connected to {rtsp_url}")
    last_print = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if time.time() - last_print > 1.0:
            md = meta.get()
            if md and "yolo" in md:
                y = md["yolo"]
                print(f"Frame {md.get('frame_id')}: "
                      f"{y.get('count',0)} detections "
                      f"({y.get('detection_time',0)*1000:.0f} ms)")
            last_print = time.time()
        if not args.no_display:
            cv2.imshow("RTSP Stream (Buffered)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

