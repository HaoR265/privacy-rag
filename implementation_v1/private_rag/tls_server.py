"""Direct TLS 1.3 ASGI ingress with actual peer-certificate identity.

No reverse-proxy identity header is trusted. This small HTTP/1.1 listener serves
management only; compute requires its own independent authenticated connection.
"""
from __future__ import annotations
import asyncio
import hashlib
import ssl
from urllib.parse import unquote
import h11
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MAX_MANAGEMENT_BODY = 64 * 1024 * 1024


def tls_context(cert: str, key: str, ca: str, *, server: bool) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    if not server:
        context.check_hostname = True
    context.set_ecdh_curve('X25519')
    context.load_verify_locations(cafile=ca)
    context.load_cert_chain(certfile=cert, keyfile=key)
    context.set_alpn_protocols(['http/1.1'])
    return context


async def serve(app, host: str, port: int, context: ssl.SSLContext):
    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        conn = h11.Connection(h11.SERVER, max_incomplete_event_size=16384)
        try:
            tls = writer.get_extra_info('ssl_object')
            if tls is None or tls.version() != 'TLSv1.3':
                return
            der = tls.getpeercert(binary_form=True)
            certificate = x509.load_der_x509_certificate(der)
            if not isinstance(certificate.public_key(), Ed25519PublicKey):
                return
            peer = {'certificate_der': der, 'fingerprint_sha256': hashlib.sha256(der).hexdigest()}
            request = None
            body = bytearray()
            while True:
                event = conn.next_event()
                if event is h11.NEED_DATA:
                    data = await asyncio.wait_for(reader.read(65536), 180)
                    if not data:
                        return
                    conn.receive_data(data)
                    continue
                if isinstance(event, h11.ConnectionClosed):
                    return
                if isinstance(event, h11.Request):
                    request = event
                elif isinstance(event, h11.Data):
                    body.extend(event.data)
                    if len(body) > MAX_MANAGEMENT_BODY:
                        return
                elif isinstance(event, h11.EndOfMessage):
                    break
                else:
                    return
            if request is None:
                return
            target = request.target.split(b'?', 1)
            raw_path = target[0]
            # Identity is injected only from the authenticated transport object.
            scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                     'http_version': '1.1', 'method': request.method.decode('ascii'),
                     'scheme': 'https', 'path': unquote(raw_path.decode('ascii')),
                     'raw_path': raw_path, 'query_string': target[1] if len(target)>1 else b'',
                     'root_path': '', 'headers': list(request.headers),
                     'client': writer.get_extra_info('peername'), 'server': writer.get_extra_info('sockname'),
                     'tls_peer': peer}
            delivered = False
            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                await asyncio.Event().wait()
            async def send(message):
                if message['type'] == 'http.response.start':
                    headers = list(message.get('headers', [])) + [(b'connection', b'close')]
                    writer.write(conn.send(h11.Response(status_code=message['status'], headers=headers)))
                elif message['type'] == 'http.response.body':
                    writer.write(conn.send(h11.Data(data=message.get('body', b''))))
                    if not message.get('more_body', False):
                        writer.write(conn.send(h11.EndOfMessage()))
                await writer.drain()
            await app(scope, receive, send)
        except (Exception, asyncio.CancelledError):
            # No exception/input logging: malformed HTTP and public disconnects
            # close the authenticated connection without echoing secret payloads.
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ssl.SSLError):
                pass
    return await asyncio.start_server(connection, host, port, ssl=context, ssl_handshake_timeout=30)
