"""The UDP over stream relay keeps datagram boundaries in both directions. No server needed:

    python3 tests/test_relay.py -v
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import udp_tcp_relay  # noqa: E402


def free_port(kind):
    s = socket.socket(socket.AF_INET, kind)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestRelay(unittest.TestCase):
    def test_round_trip_keeps_datagrams(self):
        # echo server standing in for teeworlds_srv
        echo = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        echo.bind(("127.0.0.1", 0))

        def run_echo():
            while True:
                data, addr = echo.recvfrom(65535)
                echo.sendto(b"echo:" + data, addr)
        threading.Thread(target=run_echo, daemon=True).start()

        folder = tempfile.mkdtemp(prefix="twrl-relay-")
        sock_path = os.path.join(folder, "private", "relay.sock")
        server_args = argparse.Namespace(listen_unix=sock_path, listen_tcp=None, udp="127.0.0.1:%d" % echo.getsockname()[1])
        threading.Thread(target=udp_tcp_relay.run_server, args=(server_args,), daemon=True).start()
        for _ in range(50):
            if os.path.exists(sock_path):
                break
            time.sleep(0.02)
        self.assertEqual(os.stat(os.path.dirname(sock_path)).st_mode & 0o777, 0o700)

        # stands in for `ssh -L tcp_port:sock_path`
        tcp_port = free_port(socket.SOCK_STREAM)
        listener = socket.socket()
        listener.bind(("127.0.0.1", tcp_port))
        listener.listen(1)

        def forward():
            conn, _ = listener.accept()
            unix = socket.socket(socket.AF_UNIX)
            unix.connect(sock_path)

            def pipe(a, b):
                while True:
                    data = a.recv(65536)
                    if not data:
                        return
                    b.sendall(data)
            threading.Thread(target=pipe, args=(conn, unix), daemon=True).start()
            pipe(unix, conn)
        threading.Thread(target=forward, daemon=True).start()

        udp_port = free_port(socket.SOCK_DGRAM)
        client = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "..", "scripts", "udp_tcp_relay.py"), "client",
                                   "--listen-udp", "127.0.0.1:%d" % udp_port, "--connect-tcp", "127.0.0.1:%d" % tcp_port],
                                  stdout=subprocess.DEVNULL)
        try:
            time.sleep(0.5)
            game = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            game.settimeout(2)
            # a burst of different sizes, sent faster than they are read, must come back one by one
            payloads = [bytes([i % 256]) * size for i, size in enumerate((1, 1400, 7, 600, 0, 1200, 3) * 20)]
            for p in payloads:
                game.sendto(p, ("127.0.0.1", udp_port))
            received = [game.recvfrom(65535)[0] for _ in payloads]
            self.assertEqual(sorted(received), sorted(b"echo:" + p for p in payloads))
        finally:
            client.terminate()
            client.wait()

    def test_refuses_a_socket_in_use(self):
        folder = tempfile.mkdtemp(prefix="twrl-relay-")
        sock_path = os.path.join(folder, "relay.sock")
        args = argparse.Namespace(listen_unix=sock_path, listen_tcp=None, udp="127.0.0.1:9")
        threading.Thread(target=udp_tcp_relay.run_server, args=(args,), daemon=True).start()
        for _ in range(50):
            if udp_tcp_relay.socket_in_use(sock_path):
                break
            time.sleep(0.02)
        self.assertTrue(udp_tcp_relay.socket_in_use(sock_path))
        second = subprocess.run([sys.executable, "-I", os.path.join(HERE, "..", "scripts", "udp_tcp_relay.py"), "server",
                                 "--listen-unix", sock_path, "--udp", "127.0.0.1:9"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already listening", second.stderr)
        self.assertTrue(udp_tcp_relay.socket_in_use(sock_path))


if __name__ == "__main__":
    unittest.main()
