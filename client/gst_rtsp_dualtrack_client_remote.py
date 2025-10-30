#!/usr/bin/env python3
import argparse
import json
import sys
import threading
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

# Optional OpenCV import for overlay / display
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

Gst.init(None)


class DualTrackRtspClient:
    def __init__(self, url: str, show_video: bool = True, overlay: bool = False, latency: int = 200):
        self.url = url
        self.show_video = show_video and HAS_CV2
        self.overlay = overlay and HAS_CV2
        self.latency = latency

        self.pipeline = None
        self.meta_sink = None
        self.video_sink = None

        # Shared state between threads/callbacks
        self.latest_meta = None
        self.lock = threading.Lock()
        self.running = True

    def _build_pipeline(self):
        # Build a pipeline that pulls both RTP tracks from one RTSP session:
        #  - video (H264) -> decode -> videoconvert -> appsink (videosink)
        #  - metadata (application/GST) -> rtpgstdepay -> appsink (metasink)
        video_branch = "rtph264depay ! avdec_h264 ! videoconvert ! appsink name=videosink emit-signals=true sync=false drop=true max-buffers=1"
        if not self.show_video:
            video_branch = "fakesink sync=false"

        launch = (
            f"rtspsrc location={self.url} latency={self.latency} name=src "
            f"src. ! application/x-rtp,media=video,encoding-name=H264 ! {video_branch} "
            f"src. ! application/x-rtp,media=application,encoding-name=GST ! rtpgstdepay ! "
            f"appsink name=metasink emit-signals=true sync=false drop=true max-buffers=16"
        )

        self.pipeline = Gst.parse_launch(launch)

        # Grab appsinks
        self.meta_sink = self.pipeline.get_by_name("metasink")
        if self.meta_sink:
            self.meta_sink.connect("new-sample", self._on_new_meta_sample)

        if self.show_video:
            self.video_sink = self.pipeline.get_by_name("videosink")
            if self.video_sink:
                self.video_sink.connect("new-sample", self._on_new_video_sample)

        # Add bus watch to gracefully handle errors/EOS
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[GStreamer ERROR] {err}", file=sys.stderr)
            if dbg:
                print(dbg, file=sys.stderr)
            self.running = False
        elif t == Gst.MessageType.EOS:
            print("[GStreamer] End of stream", file=sys.stderr)
            self.running = False

    def _on_new_meta_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            data = bytes(mapinfo.data)
            # Try JSON parse; if it fails, print raw bytes first 64
            try:
                meta = json.loads(data.decode("utf-8"))
                # Print pretty JSON to stdout
                print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
                # Cache for optional overlay
                with self.lock:
                    self.latest_meta = meta
            except Exception as e:
                # Fallback: show a brief hexdump prefix
                preview = data[:64]
                print(json.dumps({"non_json_payload_prefix": list(preview)}), flush=True)
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def _on_new_video_sample(self, sink):
        if not self.show_video:
            return Gst.FlowReturn.OK

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value("width")
        height = s.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
            # Assuming videoconvert output defaults to BGRx or BGR – try to infer
            # Many builds output BGR by default; if not, fallback to BGRx reshape.
            expected = width * height * 3
            if frame.size == expected:
                frame = frame.reshape((height, width, 3))
            else:
                # Try BGRx
                expected4 = width * height * 4
                if frame.size == expected4:
                    frame = frame.reshape((height, width, 4))
                    # Drop alpha channel
                    frame = frame[:, :, :3]
                else:
                    # Unsupported format; skip rendering
                    return Gst.FlowReturn.OK

            if self.overlay:
                with self.lock:
                    meta = self.latest_meta.copy() if self.latest_meta else None
                if meta:
                    y = meta.get("yolo", {})
                    dets = y.get("detections", [])
                    for d in dets:
                        bbox = d.get("bbox", None)
                        cls = d.get("class", "obj")
                        conf = d.get("conf", 0.0)
                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{cls}:{conf:.2f}", (x1, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("RTSP Video (dual-track)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def run(self):
        self._build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)

        loop = GLib.MainLoop()
        # Run the GLib main loop in a thread so Ctrl-C works to stop video window
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            if HAS_CV2 and self.show_video:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="RTSP URL carrying video + metadata tracks")
    ap.add_argument("--overlay", action="store_true", help="Draw detections on video (default off)")
    ap.add_argument("--no-video", action="store_true", help="Do not display video window")
    ap.add_argument("--latency", type=int, default=200, help="RTSP jitterbuffer latency (ms)")
    args = ap.parse_args()

    client = DualTrackRtspClient(
        url=args.url,
        show_video=(not args.no-video),
        overlay=args.overlay,
        latency=args.latency,
    )
    client.run()


if __name__ == "__main__":
    main()
