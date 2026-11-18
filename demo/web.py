"""網頁版 Demo 畫面（老師建議 1、5：即時呈現「自然語言 → Agent 動作 → 飛控指令 → 執行結果」，UI 要清楚）。

只用標準函式庫（http.server + Server-Sent Events），前端是單一 HTML 檔、不連外部網路，Demo 當天在學校網路也能跑。

    GET  /           畫面（demo/web_static/index.html）
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

from demo.web_state import build_state

STATIC = Path(__file__).with_name("web_static")


class WebDemo:
    def __init__(self, engine, speed: float = 1.0, backend: str = "sim", link=None,
                 scripted: Optional[List[Tuple[float, str]]] = None, on_step: Optional[Callable] = None,
                 duration_s: Optional[float] = None):
        self.engine = engine
        self.speed = speed
        self.backend = backend
        self.link = link
        self.scripted = list(scripted or [])
        self.on_step = on_step  # 每一步之後呼叫（終端機印事件用）
        self.duration_s = duration_s
        self._lock = threading.Lock()
        self._state_json = json.dumps(build_state(engine, backend, link), ensure_ascii=False)
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
                    state = json.dumps(build_state(eng, self.backend, self.link), ensure_ascii=False)
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
                if self.path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    self._send(200, demo.state()[1].encode("utf-8"), "application/json; charset=utf-8")
                elif self.path == "/events":
                    self._events()
                else:
                    self._send(404, b"not found", "text/plain")

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

        srv = ThreadingHTTPServer((host, port), Handler)
        srv.daemon_threads = True
        return srv


def run_web(engine, speed: float, scripted, port: int = 8000, backend: str = "sim", link=None,
            open_browser: bool = True, on_step=None, duration_s: Optional[float] = None) -> None:
    """啟動網頁 Demo，直到 Ctrl+C（或跑滿 duration_s）。"""
    demo = WebDemo(engine, speed, backend, link, scripted, on_step, duration_s).start()
    srv = demo.make_server(port=port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
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
