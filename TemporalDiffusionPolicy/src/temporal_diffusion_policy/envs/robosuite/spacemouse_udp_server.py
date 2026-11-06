import argparse
import platform
import subprocess
import time
import hid
import socket
import struct
import sys
import threading

# New-NetFirewallRule -DisplayName "Allow WSL Port 26211 UDP" -Direction Inbound -Protocol UDP -LocalPort 26211 -Action Allow

def find_spacemouse():
    """
    Search for a SpaceMouse HID device and return an opened device handle.
    Raises RuntimeError if no device is found.
    """
    for d in hid.enumerate():
        if (d['product_string'] and 'spacemouse' in d['product_string'].lower()) or (d['vendor_id'] == 9583):
            # Alternatively: check vendor/product ID instead of string match
            device = hid.device()
            device.open_path(d['path'])
            print("SpaceMouse connected!")
            return device
    raise RuntimeError("No SpaceMouse found")

def read_data_and_format(device: hid.device):
    """
    Read a single HID report from the SpaceMouse and return it packed as bytes.
    Pads to 13 bytes if the report is shorter (for newer models).
    """
    report = device.read(13, timeout_ms=100)  # SpaceMouse reports are usually 13 bytes
    if report:
        report += [0] * (13 - len(report))  # pad if needed
        return struct.pack('13B', *report)
    return None

def client_listener(sock: socket.socket, clients: set, lock: threading.Lock, last_seen: dict):
    """Thread that listens for new client packets and registers them."""
    while True:
        try:
            _, client_addr = sock.recvfrom(1024)
            with lock:
                if client_addr not in clients:
                    clients.add(client_addr)
                    print(f"Client {client_addr} connected")
                last_seen[client_addr] = time.time()
        except Exception:
            continue

def client_reaper(clients, lock, last_seen, timeout=35.0, interval=1.0):
    """Remove clients that haven't pinged within timeout seconds."""
    while True:
        time.sleep(interval)
        now = time.time()
        dead = []
        with lock:
            for client, last in list(last_seen.items()):
                if now - last > timeout:
                    print(f"Client {client} timed out")
                    dead.append(client)
            for dc in dead:
                clients.discard(dc)
                last_seen.pop(dc, None)

def serve(device: hid.device, host: str, port: int):
    """
    Start a UDP server that streams HID data to all clients that send at least one packet.
    New clients can connect at any time.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((host, port))
            print(f"Server listening on {host}:{port}")

            clients = set()
            last_seen = {}
            lock = threading.Lock()
            threading.Thread(target=client_listener, args=(s, clients, lock, last_seen), daemon=True).start()
            threading.Thread(target=client_reaper, args=(clients, lock, last_seen), daemon=True).start()

            while True:
                # Read HID data and broadcast to all clients
                payload = read_data_and_format(device)
                if payload:
                    with lock:
                        for client in list(clients):
                            try:
                                s.sendto(payload, client)
                            except OSError:
                                print(f"Send failed, dropping {client}")
                                clients.discard(client)
                                last_seen.pop(client, None)


    except KeyboardInterrupt:
        print("\nServer shutting down...")

    finally:
        print("Closing HID device...")
        try:
            device.close()
        except Exception as e:
            print("Error closing HID device:", e)
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream SpaceMouse HID reports over UDP")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=26211, help="UDP port to bind (default: 26211)")
    args = parser.parse_args()

    spacemouse = find_spacemouse()
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/F", "/IM", "3DxService.exe"], shell=True)
    else:
        print("Not running on Windows, skipping taskkill command.")
    serve(spacemouse, args.host, args.port)