#!/usr/bin/env python3
"""
YOLO RTSP Server - WITH SYNCHRONIZED METADATA
Adds presentation timestamps to align metadata with video frames
"""

import cv2
from ultralytics import YOLO
import subprocess
import threading
import time
from datetime import datetime
import argparse
import json
import socket
import struct
import queue

try:
    import gpsd
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False


class VideoSource:
    def __init__(self, source):
        print(f"Opening: {source}")
        if source.startswith('/dev/'):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        time.sleep(1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open: {source}")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read from: {source}")
        
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or self.fps > 60:
            self.fps = 25
        
        print(f"✓ Video: {self.width}x{self.height} @ {self.fps}fps")
    
    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        self.cap.release()


class YOLODetector:
    def __init__(self, model_name, conf=0.5):
        print(f"Loading YOLO: {model_name}")
        self.model = YOLO(model_name)
        self.conf = conf
        self.detection_queue = queue.Queue(maxsize=2)
        self.result_dict = {}  # frame_id -> detections
        self.running = True
        
        # Start detection thread
        threading.Thread(target=self._worker, daemon=True).start()
        print(f"✓ YOLO loaded")
    
    def _worker(self):
        """Background thread for YOLO detection"""
        while self.running:
            try:
                frame_data = self.detection_queue.get(timeout=1)
                frame_id = frame_data['id']
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                # Run detection
                start_time = time.time()
                results = self.model(frame, conf=self.conf, verbose=False)
                detection_time = time.time() - start_time
                
                detections = []
                for result in results:
                    for box in result.boxes:
                        detections.append({
                            'class': result.names[int(box.cls)],
                            'conf': round(float(box.conf), 2),
                            'bbox': [int(x) for x in box.xyxy[0]]
                        })
                
                # Store with frame ID and timestamp
                self.result_dict[frame_id] = {
                    'detections': detections,
                    'timestamp': timestamp,
                    'detection_time': detection_time,
                    'frame_id': frame_id
                }
                
                # Keep only last 100 results
                if len(self.result_dict) > 100:
                    oldest = min(self.result_dict.keys())
                    del self.result_dict[oldest]
                    
            except queue.Empty:
                continue
    
    def queue_frame(self, frame, frame_id, timestamp):
        """Queue frame for detection"""
        try:
            self.detection_queue.put_nowait({
                'id': frame_id,
                'frame': frame,
                'timestamp': timestamp
            })
        except queue.Full:
            pass
    
    def get_detection_for_frame(self, frame_id):
        """Get detection results for specific frame"""
        return self.result_dict.get(frame_id)
    
    def stop(self):
        self.running = False


class GPSReader:
    def __init__(self):
        self.data = {}
        self.enabled = GPS_AVAILABLE
        if self.enabled:
            threading.Thread(target=self._read, daemon=True).start()
    
    def _read(self):
        try:
            gpsd.connect()
            print("✓ GPS connected")
        except Exception as e:
            print(f"⚠ GPS failed: {e}")
            self.enabled = False
            return
        
        while True:
            try:
                packet = gpsd.get_current()
                self.data = {k: v for k, v in {
                    'lat': getattr(packet, 'lat', None),
                    'lon': getattr(packet, 'lon', None),
                    'alt': getattr(packet, 'alt', None),
                    'speed': getattr(packet, 'hspeed', None),
                }.items() if v is not None}
            except:
                time.sleep(1)
    
    def get(self):
        return self.data if self.enabled else {}


class RTSPServer:
    """RTSP server with timestamped metadata"""
    
    def __init__(self, width, height, fps, port=8554):
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.start_time = time.time()
        
        # FFmpeg RTSP server
        cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-g', str(fps * 2),
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            '-listen', '1',
            f'rtsp://0.0.0.0:{port}/live'
        ]
        
        print(f"Starting RTSP server on port {port}...")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        self.metadata_port = port + 1000
        self.meta_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        threading.Thread(target=self._monitor, daemon=True).start()
        time.sleep(3)
        
        print(f"✓ RTSP server listening on rtsp://0.0.0.0:{port}/live")
        print(f"✓ Metadata on UDP port {self.metadata_port}")
    
    def _monitor(self):
        for line in iter(self.process.stderr.readline, b''):
            line_str = line.decode('utf-8', errors='ignore').strip()
            if any(word in line_str.lower() for word in ['listening', 'error', 'failed']):
                print(f"FFmpeg: {line_str}")
    
    def get_timestamp(self):
        """Get timestamp since stream start (for PTS)"""
        return time.time() - self.start_time
    
    def write_frame(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            return True
        except:
            return False
    
    def send_metadata(self, metadata):
        """Broadcast metadata with timestamp"""
        try:
            json_bytes = json.dumps(metadata).encode('utf-8')
            packet = struct.pack('!I', len(json_bytes)) + json_bytes
            self.meta_sock.sendto(packet, ('127.0.0.1', self.metadata_port))
        except:
            pass
    
    def close(self):
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.meta_sock.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='YOLO RTSP Server with Synchronized Metadata')
    parser.add_argument('--input', required=True, help='Video source')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence')
    parser.add_argument('--detection-fps', type=int, default=5, help='Detection FPS')
    parser.add_argument('--port', type=int, default=8554, help='RTSP port')
    parser.add_argument('--disable-gps', action='store_true', help='Disable GPS')
    
    args = parser.parse_args()
    
    if not args.model.endswith('.pt'):
        args.model += '.pt'
    
    # Initialize
    print("\nInitializing...")
    video = VideoSource(args.input)
    yolo = YOLODetector(args.model, args.conf)
    gps = GPSReader() if not args.disable_gps else None
    
    # Start server
    server = RTSPServer(video.width, video.height, video.fps, args.port)
    
    print(f"\n{'='*70}")
    print(f"YOLO RTSP Server with SYNCHRONIZED Metadata")
    print(f"{'='*70}")
    print(f"RTSP Stream: rtsp://127.0.0.1:{args.port}/live")
    print(f"Metadata:    UDP port {args.port + 1000}")
    print(f"{'='*70}")
    print(f"Synchronization: Metadata includes PTS (Presentation Timestamp)")
    print(f"{'='*70}\n")
    print(f"Connect client:")
    print(f"  python rtsp_client_synced.py --port {args.port}")
    print(f"\nWaiting for client...\n")
    
    frame_count = 0
    detection_interval = 1.0 / args.detection_fps
    last_detection_frame = 0
    metadata_send_interval = 0.1  # Send metadata 10 times per second
    last_metadata_send = time.time()
    
    try:
        while True:
            frame = video.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            current_time = time.time()
            timestamp = server.get_timestamp()  # PTS
            
            # Queue frame for YOLO detection at lower FPS
            if frame_count - last_detection_frame >= (video.fps / args.detection_fps):
                yolo.queue_frame(frame.copy(), frame_count, timestamp)
                last_detection_frame = frame_count
            
            # Send metadata frequently with frame associations
            if current_time - last_metadata_send >= metadata_send_interval:
                # Build metadata with current frame info
                metadata = {
                    'utc': datetime.utcnow().isoformat() + 'Z',
                    'frame_id': frame_count,
                    'pts': timestamp,  # Presentation timestamp
                    'stream_time': timestamp
                }
                
                # Add GPS if available
                if gps:
                    gps_data = gps.get()
                    if gps_data:
                        metadata['gps'] = gps_data
                
                # Find most recent detection for this frame or earlier
                # This associates metadata with the actual analyzed frame
                best_match = None
                for fid in range(frame_count, max(0, frame_count - 50), -1):
                    detection_result = yolo.get_detection_for_frame(fid)
                    if detection_result:
                        best_match = detection_result
                        break
                
                if best_match:
                    metadata['yolo'] = {
                        'analyzed_frame_id': best_match['frame_id'],
                        'analyzed_pts': best_match['timestamp'],
                        'detection_latency': timestamp - best_match['timestamp'],
                        'detection_time': best_match['detection_time'],
                        'count': len(best_match['detections']),
                        'detections': best_match['detections']
                    }
                
                server.send_metadata(metadata)
                
                # Print diagnostic info
                if best_match and best_match['detections']:
                    lag_frames = frame_count - best_match['frame_id']
                    lag_ms = (timestamp - best_match['timestamp']) * 1000
                    classes = [d['class'] for d in best_match['detections']]
                    print(f"[Frame {frame_count}] Detections from frame {best_match['frame_id']} "
                          f"(lag: {lag_frames} frames / {lag_ms:.0f}ms): {classes}")
                
                last_metadata_send = current_time
            
            # Write frame
            if not server.write_frame(frame):
                time.sleep(0.1)
                continue
            
            time.sleep(1.0 / video.fps)
    
    except KeyboardInterrupt:
        print("\n✓ Stopping...")
    finally:
        video.release()
        yolo.stop()
        server.close()
        print(f"✓ Done: {frame_count} frames")


if __name__ == '__main__':
    main()
