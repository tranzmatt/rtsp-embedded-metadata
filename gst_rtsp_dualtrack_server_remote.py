#!/usr/bin/env python3
import argparse
import os
import json
import time
import cv2
import threading
import subprocess
import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer
from ultralytics import YOLO
import urllib.request


# ---- Optional GPSD support ----
try:
    import gps as gpsd
    HAS_GPSD = True
except Exception:
    HAS_GPSD = False

Gst.init(None)

def ensure_model_exists(model_path: str):
    if os.path.exists(model_path):
        return model_path

    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
    fname = os.path.basename(model_path)
    url = f"{base_url}/{fname}"

    print(f"[Model] {model_path} not found. Downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"[Model] Download complete: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download YOLO model from {url}: {e}")
    return model_path


def has_element(name: str) -> bool:
    try:
        res = subprocess.run(["gst-inspect-1.0", name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False

def select_encoder():
    if has_element("nvv4l2h264enc"):
        print("[Encoder] Using NVIDIA Jetson hardware encoder (nvv4l2h264enc)", file=sys.stderr)
        # Jetson encoder: safe, valid properties only
        return "nvv4l2h264enc bitrate=4000000 insert-sps-pps=true iframeinterval=30"
    elif has_element("nvh264enc"):
        print("[Encoder] Using NVIDIA NVENC hardware encoder (nvh264enc)", file=sys.stderr)
        return "nvh264enc preset=llhp bitrate=4000000 insert-sps-pps=true"
    else:
        print("[Encoder] Using software encoder (x264enc)", file=sys.stderr)
        return "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000000"

class DualTrackFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, fps, encoder_str):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder_str = encoder_str
        self.set_shared(True)
        self.launch_str = (
            f"( appsrc name=vidsrc is-live=true format=time do-timestamp=true "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
            f"! queue ! videoconvert ! {encoder_str} ! h264parse config-interval=1 ! rtph264pay name=pay0 pt=96 ) "
            f"( appsrc name=metasrc is-live=true format=time do-timestamp=true "
            f"caps=application/x-gst,rate={fps} "
            f"! queue ! rtpgstpay name=pay1 pt=98 )"
        )

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_str)

    def do_configure(self, rtsp_media):
        self.pipeline = rtsp_media.get_element()
        self.vidsrc = self.pipeline.get_by_name("vidsrc")
        self.metasrc = self.pipeline.get_by_name("metasrc")
        if self.vidsrc:
            self.vidsrc.set_property("block", False)
        if self.metasrc:
            self.metasrc.set_property("block", False)


def ns_per_frame(fps):
    return int(1e9 / fps)


# ---- GPSD polling thread ----
class GpsPoller(threading.Thread):
    def __init__(self, host="localhost", port=2947, interval=1.0):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.interval = float(interval)
        self.retry_delay = 30.0  # hardcoded per request
        self.running = True
        self.session = None
        self.fix = None
        self._last_attempt = 0.0

    def run(self):
        while self.running:
            now = time.time()
            if self.session is None and (now - self._last_attempt) >= self.retry_delay:
                self._last_attempt = now
                try:
                    self.session = gpsd.gps(host=self.host, port=self.port, mode=gpsd.WATCH_ENABLE)
                    # Do not spam stdout; connect notice to stderr only
                    print(f"[GPS] Connected to gpsd at {self.host}:{self.port}", file=sys.stderr)
                except Exception as e:
                    # Quiet retry (stderr once per attempt)
                    print(f"[GPS] gpsd not available ({e}); retrying in {self.retry_delay}s", file=sys.stderr)
                    self.session = None

            if self.session is not None:
                try:
                    report = self.session.next()
                    # TPV contains position/velocity
                    if isinstance(report, dict):
                        cls = report.get("class")
                        if cls == "TPV":
                            self.fix = {
                                "lat": report.get("lat"),
                                "lon": report.get("lon"),
                                "alt": report.get("alt"),
                                "speed": report.get("speed")
                            }
                    else:
                        # Some bindings expose attributes
                        if getattr(report, "class", None) == "TPV":
                            self.fix = {
                                "lat": getattr(report, "lat", None),
                                "lon": getattr(report, "lon", None),
                                "alt": getattr(report, "alt", None),
                                "speed": getattr(report, "speed", None)
                            }
                except StopIteration:
                    # gpsd ended; drop session and retry later
                    self.session = None
                    time.sleep(self.retry_delay)
                except Exception:
                    time.sleep(self.interval)
            else:
                time.sleep(self.retry_delay)

    def latest(self):
        return self.fix


class Producer:
    def __init__(self, input_url, model_path, fps_override=None):
        self.cap = cv2.VideoCapture(input_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open input: {input_url}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or (fps_override or 25)
        if fps_override:
            self.fps = fps_override
        model_path = ensure_model_exists(model_path)
        self.model = YOLO(model_path)
        self.frame_id = 0
        self.running = True

    def next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame_id += 1
        return frame

    def stop(self):
        self.running = False
        self.cap.release()


def build_push_pipeline(output_url, width, height, fps, encoder_str):
    host = "localhost"
    port = 8554
    pipeline_str = (
        f"appsrc name=vidsrc is-live=true format=time do-timestamp=true "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        f"! queue ! videoconvert ! {encoder_str} ! h264parse ! rtph264pay config-interval=1 pt=96 "
        f"! udpsink host={host} port={port} sync=false"
    )
    print('[Pipeline]', pipeline_str)
    return pipeline_str

def build_push_pipeline_1(output_url, width, height, fps, encoder_str):
    pipeline_str = (
        f"appsrc name=vidsrc is-live=true format=time do-timestamp=true "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        f"! queue ! videoconvert ! {encoder_str} ! h264parse ! rtph264pay name=pay0 pt=96 "
        f"appsrc name=metasrc is-live=true format=time do-timestamp=true "
        f"caps=application/x-gst,rate={fps} "
        f"! queue ! rtpgstpay name=pay1 pt=98 "
        f"rtpbin name=rtpbin "
        f"rtpbin.send_rtp_sink_0 :: pay0.src rtpbin.send_rtp_sink_1 :: pay1.src "
        f"rtspclientsink location={output_url}"
    )
    return pipeline_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video/camera URL (RTSP/HTTP/file)")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--fps", type=int, default=0, help="Override output FPS if non-zero")
    ap.add_argument("--port", type=int, default=8554, help="RTSP TCP port to listen on (server mode)")
    ap.add_argument("--path", default="/live", help="RTSP mount point path (server mode)")
    ap.add_argument("--output", help="Optional remote RTSP URL to push (client mode)")
    # ---- GPS args ----
    ap.add_argument("--gpsd", action="store_true", help="Enable GPSD integration")
    ap.add_argument("--gps-host", default="localhost", help="GPSD host (default: localhost)")
    ap.add_argument("--gps-port", type=int, default=2947, help="GPSD port (default: 2947)")
    ap.add_argument("--gps-interval", type=float, default=1.0, help="GPS polling interval seconds (default: 1.0)")
    # Optional printing
    ap.add_argument("--print-detections", action="store_true", help="Also print detection JSON to stdout")
    args = ap.parse_args()

    encoder_str = select_encoder()
    producer = Producer(args.input, args.model, fps_override=(args.fps or None))

    # Start GPS thread if requested and available
    gps_thread = None
    if args.gpsd and HAS_GPSD:
        gps_thread = GpsPoller(host=args.gps_host, port=args.gps_port, interval=args.gps_interval)
        gps_thread.start()
        print(f"[GPS] Enabled via gpsd at {args.gps_host}:{args.gps_port}", file=sys.stderr)
    elif args.gpsd and not HAS_GPSD:
        print("[GPS] gps python package not available; continuing without GPS", file=sys.stderr)

    if args.output:
        print(f"[Mode] Pushing multiplexed RTSP stream to {args.output}", file=sys.stderr)
        pipeline_str = build_push_pipeline(args.output, producer.width, producer.height, producer.fps, encoder_str)
        pipeline = Gst.parse_launch(pipeline_str)
        vidsrc = pipeline.get_by_name("vidsrc")
        metasrc = pipeline.get_by_name("metasrc")
        pipeline.set_state(Gst.State.PLAYING)
    else:
        print(f"[Mode] Local RTSP server at rtsp://0.0.0.0:{args.port}{args.path}", file=sys.stderr)
        loop = GLib.MainLoop()
        server = GstRtspServer.RTSPServer()
        server.props.service = str(args.port)
        factory = DualTrackFactory(producer.width, producer.height, producer.fps, encoder_str)
        server.get_mount_points().add_factory(args.path, factory)
        server.attach(None)
        vidsrc = None
        metasrc = None
        def on_media_configure(factory, media):
            pipeline = media.get_element()
            nonlocal vidsrc, metasrc
            vidsrc = pipeline.get_by_name("vidsrc")
            metasrc = pipeline.get_by_name("metasrc")
        factory.connect("media-configure", on_media_configure)
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()

    ns_frame = ns_per_frame(producer.fps)
    try:
        while producer.running:
            frame = producer.next_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            pts = producer.frame_id * ns_frame

            t0 = time.time()
            results = producer.model(frame, conf=0.5, verbose=False)
            t_det = time.time() - t0

            dets = []
            for r in results:
                names = r.names
                for b in r.boxes:
                    cls = int(b.cls)
                    dets.append({
                        "class": names.get(cls, str(cls)),
                        "conf": float(b.conf),
                        "bbox": [int(x) for x in b.xyxy[0]]
                    })

            # GPS fix (top-level), or None
            gps_fix = gps_thread.latest() if gps_thread else None

            meta = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                "frame_id": producer.frame_id,
                "pts_ns": int(pts),
                "yolo": {
                    "analyzed_frame_id": producer.frame_id,
                    "analyzed_pts_ns": int(pts),
                    "detection_time_ms": int(t_det * 1000),
                    "count": len(dets),
                    "detections": dets
                },
                "gps": gps_fix
            }
            payload = json.dumps(meta).encode("utf-8")

            if metasrc:
                mbuf = Gst.Buffer.new_allocate(None, len(payload), None)
                mbuf.fill(0, payload)
                mbuf.pts = pts
                mbuf.dts = pts
                mbuf.duration = ns_frame
                metasrc.emit("push-buffer", mbuf)

            if vidsrc:
                buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
                buf.fill(0, frame.tobytes())
                buf.pts = pts
                buf.dts = pts
                buf.duration = ns_frame
                vidsrc.emit("push-buffer", buf)

            if args.print_detections:
                # Compact JSON on server stdout to avoid bloating logs
                print(json.dumps(meta, separators=(',', ':')))
                if gps_fix is not None:
                    print(f"[GPS] lat={gps_fix.get('lat')} lon={gps_fix.get('lon')} alt={gps_fix.get('alt')} speed={gps_fix.get('speed')}")

            # Frame pacing
            time.sleep(max(0.0, (1.0 / producer.fps) - (time.time() - t0)))

    except KeyboardInterrupt:
        pass
    finally:
        producer.stop()
        if args.output:
            pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
