# server.py
import asyncio
import time
import struct
import cv2
import numpy as np
import websockets

# SETTINGS
HOST = '0.0.0.0'
PORT = 8765
TARGET_FPS = 30
JPEG_QUALITY = 70
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

def pack_timestamp_ms():
    ts_ms = time.time() * 1000.0      # correct: absolute time in ms
    return struct.pack("<d", ts_ms)   # little-endian float64

async def producer_handler(websocket):
    print("Client connected:", websocket.remote_address)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    frame_interval = 1.0 / TARGET_FPS

    # ---- SERVER FPS COUNTER ----
    frame_times = []

    try:
        while True:
            start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # simple processing example
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            proc = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # ---- JPEG encode ----
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            success, jpg = cv2.imencode('.jpg', proc, encode_param)
            if not success:
                continue

            jpg_bytes = jpg.tobytes()

            # ---- UPDATE SERVER FPS ----
            now = time.monotonic()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t <= 1.0]
            server_fps = len(frame_times)

            # ---- HEADER ----
            # send ONLY timestamp (8 bytes)
            header = pack_timestamp_ms()

            payload = header + jpg_bytes

            await websocket.send(payload)

            # ---- FPS LIMIT ----
            elapsed = time.monotonic() - start
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
    async with websockets.serve(
        producer_handler,
        HOST,
        PORT,
        max_size=None
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
