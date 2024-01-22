import json
import os
import sys

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

class ServiceSimulator:
    def __init__(self, handler):
        self.__server = ThreadingHTTPServer(("localhost", 0), handler)
        self.__thread = Thread(target=self.__server.serve_forever, daemon=True)

    @staticmethod
    def arg_pattern_is_present(pattern):
        return any(pattern in v for v in os.environ.values()) or any(pattern in v for v in sys.argv)

    @staticmethod
    def arg_pattern_replace(pattern, replacement):
        for key, value in list(os.environ.items()):
            if pattern in value:
                os.environ[key] = value.replace(pattern, replacement)

        for i, arg in list(enumerate(sys.argv)):
            if pattern in arg:
                sys.argv[i] = arg.replace(pattern, replacement)

    @property
    def host(self):
        name, port = self.__server.server_address
        return f"http://{name}:{port}"

    def start(self):
        self.__thread.start()

    def stop(self):
        self.__server.shutdown()

    class RequestHandler(BaseHTTPRequestHandler):
        def log_request(self, *args, **kwargs):
            pass

        def send_json_response(self, body, *, code=200):
            body = json.dumps(body).encode("utf-8")

            self.send_response(code)
            self.send_header("Content-Length", len(body))
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            self.wfile.write(body)
