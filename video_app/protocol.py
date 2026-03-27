"""Protocole TCP : ligne d’identification puis trames JPEG préfixées par la longueur."""

import struct
from typing import BinaryIO

HEADER_MAX = 256


def read_line(sock) -> str:
    buf = bytearray()
    while len(buf) < HEADER_MAX:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("connexion fermée pendant l’en-tête")
        if b == b"\n":
            return buf.decode("utf-8", errors="replace").strip()
        buf.extend(b)
    raise ValueError("ligne d’en-tête trop longue")


def send_camera_header(sock, name: str) -> None:
    line = f"CAMERA {name}\n".encode("utf-8")
    sock.sendall(line)


def recv_exact(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        data = sock.recv(remaining)
        if not data:
            raise ConnectionError("fin de flux")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def recv_jpeg_frame(sock) -> bytes:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length > 50 * 1024 * 1024:
        raise ValueError(f"trame JPEG anormalement grande: {length}")
    return recv_exact(sock, length)


def send_jpeg_frame(sock, jpeg: bytes) -> None:
    sock.sendall(struct.pack(">I", len(jpeg)) + jpeg)
