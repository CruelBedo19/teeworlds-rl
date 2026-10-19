#!/usr/bin/env python3
"""Carry a Teeworlds client's UDP traffic over a stream connection such as an SSH forward.

Every datagram travels as a 2 byte big endian length followed by the data, so datagram
boundaries survive the stream. Standard library only, Python 3.6 or newer, so the client
side runs on a stock macOS python3.

On the box (next to the server, which listens on 127.0.0.1 only):
    udp_tcp_relay.py server --listen-unix ~/.twrl-watch/relay.sock --udp 127.0.0.1:<sv_port>

On the Mac:
    ssh -N -L 127.0.0.1:8304:/home/<user>/.twrl-watch/relay.sock <remote-host>
    udp_tcp_relay.py client --listen-udp 127.0.0.1:8303 --connect-tcp 127.0.0.1:8304
    then connect the Teeworlds client to localhost:8303
"""

import argparse
import os
import socket
import struct
import sys
import threading

FRAME = struct.Struct(">H")
MAX_DATAGRAM = 65535


def log(msg):
    print("relay: " + msg, flush=True)


def parse_addr(text):
    host, port = text.rsplit(":", 1)
    return host, int(port)


def require_loopback(host, what):
    if host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("refusing to " + what + " on non-loopback address " + host)


def recv_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_frames(stream, on_datagram):
    """Call on_datagram for every framed datagram until the stream ends."""
    while True:
        head = recv_exact(stream, FRAME.size)
        if head is None:
            return
        data = recv_exact(stream, FRAME.unpack(head)[0])
        if data is None:
            return
        on_datagram(data)


def send_frame(stream, lock, data):
    with lock:
        stream.sendall(FRAME.pack(len(data)) + data)


# -- box side --

def serve_connection(conn, target):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.connect(target)
    udp.settimeout(1.0)
    lock = threading.Lock()
    done = threading.Event()

    def udp_to_stream():
        while not done.is_set():
            try:
                data = udp.recv(MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                send_frame(conn, lock, data)
            except OSError:
                break
        done.set()

    thread = threading.Thread(target=udp_to_stream, daemon=True)
    thread.start()
    try:
        read_frames(conn, udp.send)
    except OSError:
        pass
    finally:
        done.set()
        thread.join()
        udp.close()
        conn.close()
        log("client disconnected")


def run_server(args):
    target = parse_addr(args.udp)
    require_loopback(target[0], "forward")
    if args.listen_unix:
        path = os.path.abspath(os.path.expanduser(args.listen_unix))
        folder = os.path.dirname(path)
        # only this user may reach the relay; the socket file alone is not enough on every system
        os.makedirs(folder, mode=0o700, exist_ok=True)
        os.chmod(folder, 0o700)
        if os.path.exists(path):
            os.unlink(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        os.chmod(path, 0o600)
        where = path
    else:
        host, port = parse_addr(args.listen_tcp)
        require_loopback(host, "listen")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        where = "%s:%d" % (host, port)
    listener.listen(4)
    log("listening on %s, forwarding to udp %s:%d" % (where, target[0], target[1]))
    try:
        while True:
            conn, _ = listener.accept()
            log("client connected")
            threading.Thread(target=serve_connection, args=(conn, target), daemon=True).start()
    finally:
        listener.close()
        if args.listen_unix:
            try:
                os.unlink(path)
            except OSError:
                pass


# -- Mac side --

def run_client(args):
    listen = parse_addr(args.listen_udp)
    require_loopback(listen[0], "listen")
    connect = parse_addr(args.connect_tcp)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(listen)
    log("waiting for the Teeworlds client on udp %s:%d, tunnel at %s:%d" % (listen + connect))

    state = {"stream": None, "peer": None}
    lock = threading.Lock()

    def stream_to_udp(stream):
        try:
            read_frames(stream, lambda data: udp.sendto(data, state["peer"]))
        except OSError:
            pass
        with lock:
            if state["stream"] is stream:
                state["stream"] = None
        stream.close()
        log("tunnel connection closed, will reconnect on the next packet")

    while True:
        data, peer = udp.recvfrom(MAX_DATAGRAM)
        state["peer"] = peer
        stream = state["stream"]
        if stream is None:
            try:
                stream = socket.create_connection(connect, timeout=5)
                stream.settimeout(None)
            except OSError as e:
                log("cannot reach the tunnel at %s:%d (%s), is ssh -L running?" % (connect + (e,)))
                continue
            with lock:
                state["stream"] = stream
            threading.Thread(target=stream_to_udp, args=(stream,), daemon=True).start()
            log("connected through the tunnel")
        try:
            send_frame(stream, lock, data)
        except OSError:
            with lock:
                if state["stream"] is stream:
                    state["stream"] = None
            stream.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    s = sub.add_parser("server", help="box side: stream in, udp to the game server")
    where = s.add_mutually_exclusive_group(required=True)
    where.add_argument("--listen-unix", help="unix socket path (its folder is made private)")
    where.add_argument("--listen-tcp", help="127.0.0.1:port, for local testing")
    s.add_argument("--udp", required=True, help="game server address, 127.0.0.1:port")
    c = sub.add_parser("client", help="Mac side: udp from the game client, stream to the tunnel")
    c.add_argument("--listen-udp", default="127.0.0.1:8303")
    c.add_argument("--connect-tcp", default="127.0.0.1:8304")
    args = ap.parse_args()
    try:
        if args.mode == "server":
            run_server(args)
        elif args.mode == "client":
            run_client(args)
        else:
            ap.print_help()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
