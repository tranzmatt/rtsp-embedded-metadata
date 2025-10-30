#!/usr/bin/env python3
"""
RTSP Client - Shows synchronization quality
"""

import cv2
import socket
import threading
import json
import argparse
import struct
import time


class MetadataReceiver:
    def __init__(self, port):
        self.port = port
        self.latest_metadata = {}
        self.running = True
        self.metadata_history = []  # Track timing
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(1.0)
        
        threading.Thread(target=self._receive, daemon=True).start()
        print(f"✓ Listening for metadata on UDP {port}")
    
    def _receive(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                if len(data) < 4:
                    continue
                
                length = struct.unpack('!I', data[:4])[0]
                json_data = data[4:4+length]
                metadata = json.loads(json_data.decode('utf-8'))
                metadata['received_at'] = time.time()
                
                self.latest_metadata = metadata
                self.metadata_history.append(metadata)
                
                # Keep last 100
                if len(self.metadata_history) > 100:
                    self.metadata_history.pop(0)
                
            except socket.timeout:
                continue
            except Exception:
                pass
    
    def get_latest(self):
        return self.latest_metadata.copy() if self.latest_metadata else {}
    
    def stop(self):
        self.running = False
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description='RTSP Client - Synchronized Metadata')
    parser.add_argument('--port', type=int, default=8554, help='RTSP port')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--no-display', action='store_true', help='No video')
    
    args = parser.parse_args()
    
    rtsp_url = f"rtsp://{args.host}:{args.port}/live"
    
    print(f"\n{'='*70}")
    print(f"RTSP Client - Synchronized Metadata Analysis")
    print(f"{'='*70}")
    print(f"Video:    {rtsp_url}")
    print(f"Metadata: UDP port {args.port + 1000}")
    print(f"{'='*70}\n")
    
    metadata_receiver = MetadataReceiver(args.port + 1000)
    
    print("Connecting to RTSP stream...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    for i in range(10):
        if cap.isOpened():
            break
        print(f"Waiting... ({i+1}/10)")
        time.sleep(1)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Failed to connect")
        return
    
    print("✓ Connected\n")
    print("Analyzing synchronization... Press 'q' to quit\n")
    
    frame_count = 0
    last_print = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Print metadata with sync analysis
            if current_time - last_print >= 2.0:
                metadata = metadata_receiver.get_latest()
                
                print(f"\n{'='*70}")
                print(f"CLIENT FRAME {frame_count} - METADATA SYNCHRONIZATION")
                print(f"{'='*70}")
                
                if metadata:
                    print(f"Current Frame:  {frame_count}")
                    print(f"Metadata Frame: {metadata.get('frame_id', 'N/A')}")
                    
                    if 'yolo' in metadata:
                        yolo = metadata['yolo']
                        analyzed_frame = yolo.get('analyzed_frame_id', 'N/A')
                        lag_ms = yolo.get('detection_latency', 0) * 1000
                        detection_ms = yolo.get('detection_time', 0) * 1000
                        
                        print(f"\nSYNCHRONIZATION:")
                        print(f"  Analyzed Frame:    {analyzed_frame}")
                        print(f"  Detection Latency: {lag_ms:.0f}ms")
                        print(f"  Detection Time:    {detection_ms:.0f}ms")
                        
                        print(f"\nDETECTIONS ({yolo.get('count', 0)} objects):")
                        for i, det in enumerate(yolo.get('detections', []), 1):
                            print(f"  {i}. {det['class']} (conf: {det['conf']})")
                    
                    if 'gps' in metadata:
                        gps = metadata['gps']
                        print(f"\nGPS: {gps.get('lat'):.6f}, {gps.get('lon'):.6f}")
                else:
                    print("No metadata yet...")
                
                print(f"{'='*70}")
                last_print = current_time
            
            # Display pristine video
            if not args.no_display:
                cv2.imshow('RTSP Stream (Pristine)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n✓ Stopping...")
    finally:
        cap.release()
        metadata_receiver.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        print(f"\n{'='*70}")
        print(f"SYNCHRONIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total frames: {frame_count}")
        print(f"The metadata includes:")
        print(f"  - frame_id: Which frame is currently streaming")
        print(f"  - analyzed_frame_id: Which frame YOLO actually analyzed")
        print(f"  - detection_latency: Time between analysis and current frame")
        print(f"\nThis shows EXACTLY which video frame each detection corresponds to!")


if __name__ == '__main__':
    main()
