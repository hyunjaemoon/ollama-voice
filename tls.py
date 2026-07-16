"""
Self-signed TLS certificates for serving the dashboard over LAN HTTPS.

Browsers only expose the microphone in a secure context (HTTPS or
localhost), so reaching the push-to-talk dashboard from a phone on the same
Wi-Fi requires HTTPS. This module generates and caches a self-signed
certificate whose SANs cover localhost and the machine's current LAN IP,
regenerating it when the IP changes or the certificate nears expiry.
"""

import datetime
import ipaddress
import json
import socket
import subprocess
from pathlib import Path
from typing import Tuple

CERT_DIR = Path(__file__).parent / ".certs"
CERT_FILE = CERT_DIR / "server.crt"
KEY_FILE = CERT_DIR / "server.key"
META_FILE = CERT_DIR / "meta.json"
CERT_DAYS = 825  # Apple rejects certificates valid for longer


def get_lan_ip() -> str:
    """Return this machine's LAN IP (no traffic is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_self_signed_cert() -> Tuple[Path, Path]:
    """Return (certfile, keyfile) paths, generating a fresh pair if the
    cached one is missing, near expiry, or no longer covers the LAN IP."""
    lan_ip = get_lan_ip()
    if _cached_cert_valid(lan_ip):
        return CERT_FILE, KEY_FILE
    CERT_DIR.mkdir(exist_ok=True)
    print(f"🔐 Generating self-signed certificate for {lan_ip} → {CERT_DIR}/")
    try:
        _generate_with_cryptography(lan_ip)
    except ImportError:
        _generate_with_openssl(lan_ip)
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        days=CERT_DAYS)
    META_FILE.write_text(json.dumps(
        {"lan_ip": lan_ip, "expires": expires.isoformat()}))
    return CERT_FILE, KEY_FILE


def _cached_cert_valid(lan_ip: str) -> bool:
    if not (CERT_FILE.exists() and KEY_FILE.exists() and META_FILE.exists()):
        return False
    try:
        meta = json.loads(META_FILE.read_text())
        expires = datetime.datetime.fromisoformat(meta["expires"])
        margin = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        return meta.get("lan_ip") == lan_ip and expires > margin
    except Exception:  # noqa: BLE001 - any corruption → regenerate
        return False


def _san_ips(lan_ip: str) -> list:
    return sorted({"127.0.0.1", lan_ip})


def _generate_with_cryptography(lan_ip: str) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ollama-voice")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost")]
        + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in _san_ips(lan_ip)]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=CERT_DAYS))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _generate_with_openssl(lan_ip: str) -> None:
    san = "subjectAltName=DNS:localhost," + ",".join(
        f"IP:{ip}" for ip in _san_ips(lan_ip))
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
         "-days", str(CERT_DAYS), "-subj", "/CN=ollama-voice",
         "-addext", san],
        check=True, capture_output=True,
    )
