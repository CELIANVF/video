"""Protocole TCP : ligne CAMERA puis JPEG (legacy) ou V2 (JPEG + PCM optionnel)."""

import struct

HEADER_MAX = 256

PT_VIDEO = 1
PT_AUDIO = 2


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


def send_v2_header(sock, with_audio: bool, sample_rate: int = 16000, channels: int = 1) -> None:
    sock.sendall(b"V2\n")
    if with_audio:
        sock.sendall(f"AUDIO {sample_rate} {channels}\n".encode("utf-8"))


def send_v2_packet(sock, packet_type: int, payload: bytes) -> None:
    sock.sendall(
        bytes([packet_type & 0xFF]) + struct.pack(">I", len(payload)) + payload
    )


def recv_v2_packet(sock) -> tuple[int, bytes]:
    typ_b = recv_exact(sock, 1)
    typ = typ_b[0]
    (ln,) = struct.unpack(">I", recv_exact(sock, 4))
    if ln > 50 * 1024 * 1024:
        raise ValueError(f"paquet V2 trop grand: {ln}")
    return typ, recv_exact(sock, ln)


def peel_transport(conn) -> tuple[str, object]:
    """
    Après la ligne CAMERA : détecte legacy (1er JPEG) ou V2.

    Retourne :
      ('legacy', jpeg_bytes)
      ('v2', (sr, ch) | None)  # None = pas d’audio annoncé
    """
    b0 = recv_exact(conn, 1)
    if b0 == b"V":
        line = bytearray(b"V")
        while len(line) < HEADER_MAX:
            c = conn.recv(1)
            if not c:
                raise ConnectionError("fin de flux (V2)")
            if c == b"\n":
                break
            line.extend(c)
        if line.decode("utf-8", errors="replace").strip() != "V2":
            raise ValueError("en-tête V2 attendu après V")
        nxt = read_line(conn)
        audio = None
        if nxt.upper().startswith("AUDIO "):
            parts = nxt.split()
            if len(parts) >= 3:
                audio = (int(parts[1]), int(parts[2]))
        return "v2", audio
    ln_bytes = b0 + recv_exact(conn, 3)
    (length,) = struct.unpack(">I", ln_bytes)
    if length > 50 * 1024 * 1024:
        raise ValueError(f"trame JPEG anormalement grande: {length}")
    jpeg = recv_exact(conn, length)
    return "legacy", jpeg
