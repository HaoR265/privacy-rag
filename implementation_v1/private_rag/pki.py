"""Create a new internal Ed25519 CA and distinct identities; never overwrite keys."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import ipaddress
import json
import os
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def create_local_pki(destination: Path):
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(destination, 0o700)
    now = datetime.now(timezone.utc)
    key = Ed25519PrivateKey.generate()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Private RAG internal CA')])
    ca = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1))
          .not_valid_after(now+timedelta(days=365)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
          .add_extension(x509.KeyUsage(False,False,False,False,False,True,True,None,None),critical=True).sign(key,None))
    def write(name, data):
        fd=os.open(destination/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    def pem_private(k):
        return k.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption())
    write('ca.pem',ca.public_bytes(serialization.Encoding.PEM));write('ca.key',pem_private(key))
    manifest={}
    for role in ('D','U','S-management','S-compute'):
        private=Ed25519PrivateKey.generate()
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,role)])
        usage=ExtendedKeyUsageOID.SERVER_AUTH if role.startswith('S-') else ExtendedKeyUsageOID.CLIENT_AUTH
        cert=(x509.CertificateBuilder().subject_name(name).issuer_name(subject).public_key(private.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1))
              .not_valid_after(now+timedelta(days=90)).add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
              .add_extension(x509.ExtendedKeyUsage([usage]),critical=True)
              .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'),x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False)
              .sign(key,None))
        write(role+'.pem',cert.public_bytes(serialization.Encoding.PEM));write(role+'.key',pem_private(private))
        manifest[role]={'certificate_sha256':hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()}
    write('identities.json',json.dumps(manifest,sort_keys=True,indent=2).encode())
    return manifest
