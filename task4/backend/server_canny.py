# server.py
import asyncio
import time
import struct
import cv2
import numpy as np
import websockets
import json

HOST = "0.0.0.0"
PORT = 8765

TARGET_FPS = 100
JPEG_QUALITY = 70
TARGET_WIDTH = 1200
TARGET_HEIGHT = 720


def pack_timestamp_ms():
    return struct.pack("<d", time.time() * 1000.0)


def open_capture(source):
    """source = 0 for webcam, or a filename for video"""
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        # Webcam settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 60)

    return cap


async def producer_handler(websocket):
    print("Client connected:", websocket.remote_address)

    # Default: webcam
    cap = open_capture(0)

    frame_interval = 1.0 / TARGET_FPS
    frame_times = []

    try:
        while True:
            loop_start = time.monotonic()

            # ---- LISTEN FOR COMMANDS FROM CLIENT (async) ----
            # Non-blocking check for messages
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.0001)
                data = json.loads(msg)

                # frontend sends { "cmd": "use_file", "path": "video.mp4" }
                if data.get("cmd") == "use_file":
                    new_path = data.get("path")
                    print("Switching to video:", new_path)
                    cap.release()
                    cap = open_capture(new_path)

                # frontend sends { "cmd": "use_camera" }
                elif data.get("cmd") == "use_camera":
                    print("Switching to webcam")
                    cap.release()
                    cap = open_capture(0)

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print("Command error:", e)

            # ---- READ FRAME ----
            ret, frame = cap.read()
            if not ret:
                # If video file ended → loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # Example processing (same as your code)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            proc = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Encode JPEG
            ok, jpg = cv2.imencode(".jpg", proc, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            jpg_bytes = jpg.tobytes()

            # ---- Compute SERVER FPS ----
            now = time.monotonic()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t <= 1.0]
            server_fps = len(frame_times)

            # ---- HEADER: timestamp + uint16 FPS ----
            header = pack_timestamp_ms() + struct.pack("<H", server_fps)
            payload = header + jpg_bytes

            await websocket.send(payload)

            # ---- FPS LIMIT ----
            elapsed = time.monotonic() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            else:
                await asyncio.sleep(0)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        cap.release()


async def main():
    print(f"Starting server at ws://{HOST}:{PORT}")
    async with websockets.serve(producer_handler, HOST, PORT, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
