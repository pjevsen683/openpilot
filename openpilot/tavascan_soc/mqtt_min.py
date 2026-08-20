"""Minimal MQTT 3.1.1 publisher - kun stdlib, ingen credentials, publish-only QoS0."""
import socket, struct

def _rem_len(n):
    out = b""
    while True:
        b_ = n % 128
        n //= 128
        out += bytes([b_ | (0x80 if n > 0 else 0)])
        if n == 0:
            return out

def publish(host, port, client_id, msgs, timeout=5.0, retain=True):
    """msgs: liste af (topic, payload_str)."""
    cid = client_id.encode()
    # CONNECT: protocol 'MQTT', level 4, clean session, keepalive 60
    var = b"\x00\x04MQTT\x04\x02\x00\x3c" + struct.pack("!H", len(cid)) + cid
    pkt = b"\x10" + _rem_len(len(var)) + var
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(pkt)
        ack = s.recv(4)
        if len(ack) < 4 or ack[0] != 0x20 or ack[3] != 0:
            raise OSError(f"CONNACK afvist: {ack!r}")
        for topic, payload in msgs:
            t = topic.encode()
            p = payload.encode()
            body = struct.pack("!H", len(t)) + t + p
            flags = 0x30 | (0x01 if retain else 0x00)
            s.sendall(bytes([flags]) + _rem_len(len(body)) + body)
        s.sendall(b"\xe0\x00")  # DISCONNECT
    return len(msgs)

