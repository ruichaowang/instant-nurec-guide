"""Loopback viewer for an existing Scene instance; sequential GPU access."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import math

from PIL import Image
import torch


def as_image(rgb):
    return Image.fromarray(rgb.clamp(0, 1).mul(255).round().byte().cpu().numpy())


def serve(driver, pipe, args, render, enhance, cameras):
    available = math.floor(((driver.last-driver.first)/1e6-args.start)/args.step + 1e-8) + 1
    count = min(args.frames, available)
    if count < 1:
        raise ValueError("No frames in requested interval")
    page = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    page = page.replace("FRAME_MAX", str(count-1)).replace("DIFIX_DISABLED", "disabled" if pipe is None else "")
    for field in ("yaw", "pitch", "height", "rear"):
        page = page.replace(f"INITIAL_{field.upper()}", str(getattr(args, field)))
    pool = ThreadPoolExecutor(max_workers=1)
    current = pending_index = pending = state = None

    @torch.inference_mode()
    def draw(j, angles, fix):
        nonlocal current, pending_index, pending, state
        if current != j:
            if pending_index != j:
                if pending is not None:
                    pending.cancel()
                pending = pool.submit(driver.prepare, args.start+j*args.step)
            future = pending
            pending_index = pending = None
            inputs, front = future.result()
            pending_index = j+1 if j+1<count else None
            pending = (pool.submit(driver.prepare, args.start+(j+1)*args.step)
                       if pending_index is not None else None)
            scene = driver.reconstruct(inputs)
            ordered = sorted(driver.source.camera_sensors, key=lambda k: k.canonical_order)
            frames = inputs.context[0].data.camera.labels.rgb
            strip = Image.new("RGB", (1024, 112))
            for i, name in enumerate(cameras):
                group = [str(k) for k in ordered].index(name)
                thumb = as_image(frames[4*group+3]).resize((204,112))
                strip.paste(thumb, (i*204,0))
            state = scene, front, strip
            current = j
        scene, front, strip = state
        rgb = render(scene, front, *angles)
        if fix:
            rgb = enhance(pipe, rgb, front)[0].float().permute(1,2,0)
        panel = Image.new("RGB", (1024,688))
        panel.paste(as_image(rgb))
        panel.paste(strip,(0,576))
        buf = BytesIO()
        panel.save(buf, "JPEG", quality=90)
        return buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                data, mime = page.encode(), "text/html; charset=utf-8"
            elif url.path == "/view":
                q = parse_qs(url.query)
                try:
                    j = int(q["frame"][0])
                    angles = [float(q[k][0]) for k in ("yaw","pitch","height","rear")]
                    if not 0 <= j < count or not all(map(math.isfinite, angles)):
                        raise ValueError("Invalid frame or camera")
                    data = draw(j, angles, pipe is not None and q.get("fix")==["1"])
                    mime = "image/jpeg"
                except (KeyError, ValueError) as exc:
                    self.send_error(400, str(exc))
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (ConnectionResetError, BrokenPipeError):
                pass

    try:
        with HTTPServer(("127.0.0.1", args.serve), Handler) as server:
            print(f"Viewer ready on http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    finally:
        pool.shutdown(cancel_futures=True)
