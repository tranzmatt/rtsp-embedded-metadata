#!/usr/bin/env python3
"""
RTSP Client — Client-side frame alignment (Option 2).
Maintains a small buffer of video frames and displays the one whose PTS
matches the YOLO metadata timestamp.
"""

import cv2, time, json, socket, struct, threading, argparse
from collections import deque

class MetadataReceiver:
    def __init__(self,port):
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0",port))
        self.sock.settimeout(1.0)
        self.latest=None
        threading.Thread(target=self._recv,daemon=True).start()
    def _recv(self):
        while True:
            try:
                d,_=self.sock.recvfrom(65535)
                l=struct.unpack("!I",d[:4])[0]
                js=json.loads(d[4:4+l].decode())
                self.latest=js
            except: pass
    def get(self): return self.latest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8554)
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--buffer-ms",type=int,default=800,help="client frame buffer window (ms)")
    ap.add_argument("--no-display",action="store_true")
    args=ap.parse_args()

    rtsp=f"rtsp://{args.host}:{args.port}/live"
    cap=cv2.VideoCapture(rtsp,cv2.CAP_FFMPEG)
    if not cap.isOpened(): print("❌ Cannot open stream"); return
    meta=MetadataReceiver(args.port+1000)
    print(f"✓ Connected to {rtsp}")

    frame_buf=deque(); max_len=args.buffer_ms/1000.0
    last_print=time.time()

    try:
        while True:
            ok,frame=cap.read()
            if not ok: time.sleep(0.01); continue
            pts=time.time()
            frame_buf.append((pts,frame))
            # remove old frames beyond window
            while frame_buf and pts-frame_buf[0][0]>max_len:
                frame_buf.popleft()

            m=meta.get()
            if m and "yolo" in m:
                target=m["yolo"].get("analyzed_pts",None)
                if target:
                    closest=min(frame_buf,key=lambda f:abs(f[0]- (frame_buf[0][0]+target-frame_buf[-1][0]))) if frame_buf else None
                else:
                    closest=None
                if closest:
                    cf=closest[1]
                    if not args.no_display:
                        disp=cf.copy()
                        for d in m["yolo"].get("detections",[]):
                            x1,y1,x2,y2=d["bbox"]
                            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.putText(disp,d["class"],(x1,y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                        cv2.imshow("RTSP (Aligned)",disp)
                    if time.time()-last_print>1.0:
                        print(f"Frame {m['frame_id']}: {m['yolo']['count']} objects "
                              f"({m['yolo']['detection_time']*1000:.0f} ms)")
                        last_print=time.time()
            if not args.no_display:
                if cv2.waitKey(1)&0xFF==ord('q'): break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()

