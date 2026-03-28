"""Protocole TCP : ligne CAMERA puis JPEG (legacy) ou V2 (JPEG + PCM optionnel)."""

from __future__ import annotations

import socket
import struct

HEADER_MAX = 256
_RECV_CHUNK = 65536

PT_VIDEO = 1
PT_AUDIO = 2


class SockReader:
    """Lecture TCP bufferisée (évite recv(1) répété sur les en-têtes)."""

    __slots__ = ("_sock", "_b")

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._b = bytearray()

    def _recv_more(self) -> None:
        data = self._sock.recv(_RECV_CHUNK)
        if not data:
            raise ConnectionError("fin de flux")
        self._b.extend(data)

    def read_exact(self, n: int) -> bytes:
        while len(self._b) < n:
            self._recv_more()
        out = bytes(self._b[:n])
        del self._b[:n]
        return out

    def read_line(self, max_len: int = HEADER_MAX) -> str:
        while True:
            nl = self._b.find(b"\n")
            if nl >= 0:
                if nl >= max_len:
                    raise ValueError("ligne d'en-tête trop longue")
                line = bytes(self._b[:nl])
                del self._b[: nl + 1]
                return line.decode("utf-8", errors="replace").strip()
            if len(self._b) >= max_len:
                raise ValueError("ligne d'en-tête trop longue")
            self._recv_more()


def send_camera_header(sock, name: str) -> None:
    line = f"CAMERA {name}\n".encode("utf-8")
    sock.sendall(line)


def recv_jpeg_frame(reader: SockReader) -> bytes:
    header = reader.read_exact(4)
    (length,) = struct.unpack(">I", header)
    if length > 50 * 1024 * 1024:
        raise ValueError(f"trame JPEG anormalement grande: {length}")
    return reader.read_exact(length)


def send_jpeg_frame(sock, jpeg: bytes) -> None:
    sock.sendall(struct.pack(">I", len(jpeg)) + jpeg)


def send_v2_header(sock, with_audio: bool, sample_rate: int = 16000, channels: int = 1) -> None:
    sock.sendall(b"V2\n")
    if with_audio:
        sock.sendall(f"AUDIO {sample_rate} {channels}\n".encode("utf-8"))


def send_v2_packet(sock, packet_type: int, payload: bytes) -> None:
    sock.sendall(
        bytes([packet_type & 0xFF]) + struct.pack(">I", len(payload)) + payload
    )


def recv_v2_packet(reader: SockReader) -> tuple[int, bytes]:
    typ_b = reader.read_exact(1)
    typ = typ_b[0]
    (ln,) = struct.unpack(">I", reader.read_exact(4))
    if ln > 50 * 1024 * 1024:
        raise ValueError(f"paquet V2 trop grand: {ln}")
    return typ, reader.read_exact(ln)


def peel_transport(reader: SockReader) -> tuple[str, object]:
    """
    Après la ligne CAMERA : détecte legacy (1er JPEG) ou V2.

    Retourne :
      ('legacy', jpeg_bytes)
      ('v2', (sr, ch) | None)  # None = pas d'audio annoncé
    """
    b0 = reader.read_exact(1)
    if b0 == b"V":
        rest = reader.read_line()
        line_str = ("V" + rest).strip()
        if line_str != "V2":
            raise ValueError("en-tête V2 attendu après V")
        nxt = reader.read_line()
        audio = None
        if nxt.upper().startswith("AUDIO "):
            parts = nxt.split()
            if len(parts) >= 3:
                audio = (int(parts[1]), int(parts[2]))
        return "v2", audio
    ln_bytes = b0 + reader.read_exact(3)
    (length,) = struct.unpack(">I", ln_bytes)
    if length > 50 * 1024 * 1024:
        raise ValueError(f"trame JPEG anormalement grande: {length}")
    jpeg = reader.read_exact(length)
    return "legacy", jpeg
