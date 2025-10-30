#!/usr/bin/env python3
"""
YOLO RTSP Server — Streaming mode for client-side alignment (Option 2)
The server runs YOLO asynchronously and emits metadata tagged with PTS,
without delaying the outgoing RTSP stream.
"""

import cv2, time, json, struct, socket, subprocess, threading, argparse
from datetime import datetime
from ultralytics import YOLO

class VideoSource:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {src}")
        ret, f = self.cap.read()
        if not ret: raise RuntimeError("No frame read")
        self.width, self.height = f.shape[1], f.shape[0]
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
    def read(self): ok, f = self.cap.read(); return f if ok else None
    def release(self): self.cap.release()

class YOLODetector:
    def __init__(self, model, conf=0.5):
        self.model = YOLO(model); self.conf = conf
        self.results = {}; self.running = True
        import queue; self.q = queue.Queue(maxsize=2)
        threading.Thread(target=self._worker, daemon=True).start()
    def _worker(self):
        while self.running:
            try:
                fid, frame, ts = self.q.get(timeout=1)
                t0 = time.time()
                res = self.model(frame, conf=self.conf, verbose=False)
                t_det = time.time() - t0
                dets=[]
                for r in res:
                    for b in r.boxes:
                        dets.append({
                            "class":r.names[int(b.cls)],
                            "conf":float(b.conf),
                            "bbox":[int(x) for x in b.xyxy[0]]
                        })
                self.results[fid] = {"timestamp":ts,
                                     "detection_time":t_det,
                                     "detections":dets}
            except: pass
    def queue(self,fid,frame,ts):
        try: self.q.put_nowait((fid,frame,ts))
        except: pass
    def get(self,fid): return self.results.pop(fid,None)
    def stop(self): self.running=False

class RTSPServer:
    def __init__(self,w,h,fps,port):
        self.start=time.time()
        cmd=["ffmpeg","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}","-r",str(fps),
             "-i","pipe:0","-c:v","libx264","-preset","ultrafast","-tune","zerolatency",
             "-pix_fmt","yuv420p","-g",str(fps*2),"-f","rtsp","-rtsp_transport","tcp",
             "-listen","1",f"rtsp://0.0.0.0:{port}/live"]
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        self.meta_port=port+1000
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    def pts(self): return time.time()-self.start
    def write(self,frame):
        try: self.proc.stdin.write(frame.tobytes()); self.proc.stdin.flush()
        except: pass
    def send_meta(self,m):
        js=json.dumps(m).encode(); pkt=struct.pack("!I",len(js))+js
        self.sock.sendto(pkt,("127.0.0.1",self.meta_port))
    def close(self):
        self.proc.terminate(); self.sock.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--model",default="yolov8n.pt")
    ap.add_argument("--port",type=int,default=8554)
    ap.add_argument("--conf",type=float,default=0.5)
    args=ap.parse_args()

    v=VideoSource(args.input)
    y=YOLODetector(args.model,args.conf)
    s=RTSPServer(v.width,v.height,v.fps,args.port)
    fid=0; last_meta=0
    print(f"✓ Streaming RTSP on rtsp://0.0.0.0:{args.port}/live")

    try:
        while True:
            f=v.read()
            if f is None: time.sleep(0.01); continue
            fid+=1; pts=s.pts()
            s.write(f)                                 # send immediately
            y.queue(fid,f.copy(),pts)                  # async detection
            # every 0.1s emit latest metadata
            if time.time()-last_meta>0.1:
                res=None
                for k in range(fid,fid-50,-1):
                    if k in y.results: res=y.results[k]; break
                meta={"utc":datetime.utcnow().isoformat()+"Z",
                      "frame_id":fid,"pts":pts}
                if res:
                    meta["yolo"]={"analyzed_frame_id":k,
                                  "analyzed_pts":res["timestamp"],
                                  "detection_time":res["detection_time"],
                                  "count":len(res["detections"]),
                                  "detections":res["detections"]}
                s.send_meta(meta); last_meta=time.time()
            time.sleep(1.0/v.fps)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        v.release(); y.stop(); s.close()

if __name__=="__main__":
    main()

