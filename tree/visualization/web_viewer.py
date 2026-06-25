"""自定义 Web Viewer 后端。

这个文件不关心树是怎么执行的，它只负责：
- 托管前端静态页面
- 暴露 `/api/state` 快照接口

因此可以把它理解成“行为树运行结果的一个轻量 HTTP 壳”。
"""

import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class BehaviorTreeWebViewer:
    """把快照通过一个本地 HTTP 服务暴露给浏览器页面。"""

    def __init__(self, snapshot_provider, host="127.0.0.1", port=8765, static_dir=None):
        # snapshot_provider 由主节点传入，viewer 自己不关心树逻辑，只负责对外提供 HTTP 接口。
        self.snapshot_provider = snapshot_provider
        self.host = host
        self.port = port
        self.static_dir = static_dir or os.path.join(os.path.dirname(__file__), "web")
        self._server = None
        self._thread = None

    def start(self):
        """启动后台 HTTP 服务线程。"""
        if self._server is not None:
            return
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        # HTTP 服务放到后台线程，避免阻塞 ROS2 spin 主线程。
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """关闭后台 HTTP 服务并释放端口。"""
        if self._server is None:
            return
        # shutdown 会停止 serve_forever 循环，server_close 负责释放监听端口。
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def _make_handler(self):
        # handler 在这里闭包化创建，这样既能访问 snapshot_provider，也不用额外定义全局类。
        snapshot_provider = self.snapshot_provider
        static_dir = os.path.abspath(self.static_dir)

        class ViewerHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=static_dir, **kwargs)

            def do_GET(self):
                # 前端轮询 /api/state 获取最新执行快照，其余路径都走静态文件服务。
                if self.path in ("/api/state", "/api/state/"):
                    self._send_json(snapshot_provider())
                    return
                if self.path in ("/", "/index.html"):
                    self._send_static_file("index.html")
                    return
                return super().do_GET()

            def do_HEAD(self):
                # 显式支持 HEAD，避免某些客户端探测根路径时出现 404。
                if self.path in ("/", "/index.html"):
                    self._send_static_headers("index.html")
                    return
                return super().do_HEAD()

            def log_message(self, format, *args):
                return

            def _send_json(self, payload):
                # 明确设置 utf-8 与长度，浏览器端读取会更稳定。
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_static_headers(self, filename):
                file_path = os.path.join(static_dir, filename)
                if not os.path.isfile(file_path):
                    self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                    return None

                content_type, _ = mimetypes.guess_type(file_path)
                content_type = content_type or "application/octet-stream"
                file_size = os.path.getsize(file_path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.end_headers()
                return file_path

            def _send_static_file(self, filename):
                file_path = self._send_static_headers(filename)
                if file_path is None:
                    return
                with open(file_path, "rb") as handle:
                    self.wfile.write(handle.read())

        return ViewerHandler
