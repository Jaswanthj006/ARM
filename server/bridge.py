"""BoardCheck laptop bridge.

Serves the website and relays messages between the phone app and the browser.
It does no AI work: YOLO and Llama run on the phone.

  http://<laptop>:8000          website
  ws://<laptop>:8765/phone      the phone app connects here (address is in the QR code)
  ws://<laptop>:8765/ui         the website connects here

Run:  python3 server/bridge.py
"""
import asyncio
import json
import os
import socket
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websockets

HTTP_PORT = int(os.environ.get("BC_HTTP_PORT", 8000))
WS_PORT = int(os.environ.get("BC_WS_PORT", 8765))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "data", "capture")


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/info"):
            body = json.dumps({"ip": lan_ip(), "http_port": HTTP_PORT, "ws_port": WS_PORT}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # always serve the latest page
        super().end_headers()

    def log_message(self, *args):
        pass


phone = None
uis: set = set()


async def broadcast(msg):
    for ws in list(uis):
        try:
            await ws.send(msg)
        except websockets.ConnectionClosed:
            uis.discard(ws)


async def handler(ws):
    global phone
    path = ws.request.path if hasattr(ws, "request") else ws.path
    if path.startswith("/phone"):
        if phone is not None:
            await phone.close()
        phone = ws
        print("phone connected:", ws.remote_address)
        await broadcast(json.dumps({"type": "bridge", "phone": True}))
        try:
            async for msg in ws:  # frames (binary JPEG or JSON) and results
                if isinstance(msg, bytes) and os.path.isdir(RECORD_DIR) and os.path.exists(os.path.join(RECORD_DIR, "ON")):
                    # Dataset capture: create ml/data/capture/ON to save live frames, delete it to stop.
                    with open(os.path.join(RECORD_DIR, f"f_{int(time.time() * 1000)}.jpg"), "wb") as f:
                        f.write(msg)
                await broadcast(msg)
        finally:
            if phone is ws:
                phone = None
                print("phone disconnected")
                await broadcast(json.dumps({"type": "bridge", "phone": False}))
    else:
        uis.add(ws)
        await ws.send(json.dumps({"type": "bridge", "phone": phone is not None}))
        try:
            async for msg in ws:  # commands: start / stop / capture
                if phone is not None:
                    await phone.send(msg)
        finally:
            uis.discard(ws)


async def main():
    httpd = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), partial(Handler, directory=WEB_DIR))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    ip = lan_ip()
    print(f"BoardCheck website:  http://localhost:{HTTP_PORT}")
    print(f"Phone connects to:   ws://{ip}:{WS_PORT}/phone")
    async with websockets.serve(handler, "0.0.0.0", WS_PORT, max_size=8 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
