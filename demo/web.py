"""網頁版 Demo 畫面（老師建議 1、5：即時呈現「自然語言 → Agent 動作 → 飛控指令 → 執行結果」，UI 要清楚）。

只用標準函式庫（http.server + Server-Sent Events），前端是單一 HTML 檔、不連外部網路，Demo 當天在學校網路也能跑。

    GET  /           畫面（demo/web_static/index.html）
    GET  /video      影片場景的背景影片（支援 Range，瀏覽器才能跳到指定時間）
    GET  /calibrate  地面標定工具：在影片畫面上點 ≥ 4 個地面點、輸入實際座標 → 存成 config/video_ground_<id>.json
    POST /calibration 標定結果（JSON：video_id、points）
    GET  /events     SSE：每 0.1 s 推一次最新狀態（demo.web_state.build_state）
    GET  /state      目前狀態（JSON，除錯與測試用）
    POST /command    操作員自然語言指令：{"text": "把那台車趕進捕捉區抓拍"}

模擬迴圈在背景執行緒照真實時間跑（speed 倍速，接 A 的 SITL 時固定 1 倍）；HTTP 在主執行緒。
用法見 demo/README.md：python -m demo.run --web
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlsplit

from demo.web_state import build_state

STATIC = Path(__file__).with_name("web_static")


class WebDemo:
    def __init__(self, engine, speed: float = 1.0, backend: str = "sim", link=None,
                 scripted: Optional[List[Tuple[float, str]]] = None, on_step: Optional[Callable] = None,
                 duration_s: Optional[float] = None, video: Optional[dict] = None, video_file: Optional[Path] = None,
                 ground_out: Optional[Path] = None):
        self.engine = engine
        self.speed = speed
        self.backend = backend
        self.link = link
        self.scripted = list(scripted or [])
        self.on_step = on_step  # 每一步之後呼叫（終端機印事件用）
        self.duration_s = duration_s
        self.video = video  # 給前端的影片資訊（見 web_state.build_state）
        self.video_file = Path(video_file) if video_file else None
        self.ground_out = Path(ground_out) if ground_out else None  # 標定工具存檔位置
        self._lock = threading.Lock()
        self._state_json = json.dumps(build_state(engine, backend, link, video), ensure_ascii=False)
        self._version = 0
        self._stop = threading.Event()
        self.finished = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------- 模擬迴圈 ----------
    def start(self) -> "WebDemo":
        self._thread = threading.Thread(target=self._loop, name="demo-engine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        from demo.run import feed_scripted

        eng = self.engine
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                with self._lock:
                    feed_scripted(eng, self.scripted)
                    eng.step()
                    state = json.dumps(build_state(eng, self.backend, self.link, self.video), ensure_ascii=False)
                    self._state_json, self._version = state, self._version + 1
                if self.on_step:
                    self.on_step(eng)
                if self.duration_s is not None and eng.t_ns / 1e9 >= self.duration_s:
                    break
                time.sleep(max(0.0, eng.dt / self.speed - (time.monotonic() - t0)))
        finally:
            self.finished.set()

    def state(self) -> Tuple[int, str]:
        with self._lock:
            return self._version, self._state_json

    def command(self, text: str) -> None:
        self.engine.commander.command(text)  # 執行緒安全

    # ---------- HTTP ----------
    def make_server(self, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
        demo = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 不要每個請求都印在終端機
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.path = urlsplit(self.path).path  # 去掉 ?video_id=… 之類的查詢字串
                if self.path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    self._send(200, demo.state()[1].encode("utf-8"), "application/json; charset=utf-8")
                elif self.path == "/events":
                    self._events()
                elif self.path == "/video" and demo.video_file is not None:
                    self._video()
                elif self.path == "/calibrate":
                    self._send(200, (STATIC / "calibrate.html").read_bytes(), "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")

            def _video(self):
                """mp4 串流，支援 Range（瀏覽器 seek 需要 206 Partial Content）。"""
                size = demo.video_file.stat().st_size
                start, end = 0, size - 1
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    a, _, b = rng[6:].partition("-")
                    start = int(a) if a else max(0, size - int(b))
                    end = int(b) if a and b else end
                end = min(end, size - 1)
                self.send_response(206 if rng else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with open(demo.video_file, "rb") as f:
                        f.seek(start)
                        left = end - start + 1
                        while left > 0:
                            chunk = f.read(min(1 << 20, left))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while not demo._stop.is_set():
                        v, s = demo.state()
                        if v != last:
                            self.wfile.write(b"data: " + s.encode("utf-8") + b"\n\n")
                            self.wfile.flush()
                            last = v
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 瀏覽器關掉分頁

            def do_POST(self):
                self.path = urlsplit(self.path).path
                if self.path == "/calibration":
                    self._calibration()
                    return
                if self.path != "/command":
                    self._send(404, b"not found", "text/plain")
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    text = str(body.get("text", "")).strip()
                except ValueError:
                    self._send(400, b'{"ok": false, "error": "JSON"}', "application/json")
                    return
                if not text:
                    self._send(400, '{"ok": false, "error": "指令是空的"}'.encode("utf-8"), "application/json; charset=utf-8")
                    return
                demo.command(text)
                self._send(200, json.dumps({"ok": True, "text": text}, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            def _calibration(self):
                from coord_transform.homography import GroundHomography

                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    pts = body["points"]
                    g = GroundHomography.fit([p["pixel"] for p in pts], [p["ground_ne_m"] for p in pts], str(body.get("video_id", "")))
                except (KeyError, ValueError, TypeError) as e:
                    self._send(400, json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                    return
                out = {"ok": True, "rms_error_m": g.rms_error_m, "saved": None}
                if demo.ground_out is not None and body.get("save"):
                    demo.ground_out.parent.mkdir(parents=True, exist_ok=True)
                    demo.ground_out.write_text(json.dumps({"video_id": g.video_id, "points": pts}, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
                    out["saved"] = str(demo.ground_out)
                self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        srv = ThreadingHTTPServer((host, port), Handler)
        srv.daemon_threads = True
        return srv


def run_web(engine, speed: float, scripted, port: int = 8000, backend: str = "sim", link=None,
            open_browser: bool = True, on_step=None, duration_s: Optional[float] = None, video: Optional[dict] = None,
            video_file=None, ground_out=None, path: str = "/") -> None:
    """啟動網頁 Demo，直到 Ctrl+C（或跑滿 duration_s）。"""
    demo = WebDemo(engine, speed, backend, link, scripted, on_step, duration_s, video, video_file, ground_out).start()
    srv = demo.make_server(port=port)
    url = f"http://127.0.0.1:{srv.server_address[1]}{path}"
    print(f"網頁 Demo：{url}（Ctrl+C 結束）", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    th = threading.Thread(target=srv.serve_forever, name="web", daemon=True)
    th.start()
    try:
        while not demo.finished.wait(0.2):
            pass
    except KeyboardInterrupt:
        print("\n結束中…", flush=True)
    finally:
        demo.stop()
        srv.shutdown()
        srv.server_close()
