"""mTLS-only FastAPI management plane. See tls_server.py for direct TLS ingress.

Every identity comes from the actual validated peer certificate in the ASGI
scope, never from an HTTP header or request body. No original/question/vector,
source-hit, answer-plaintext or server-share download endpoint is present.
"""
from __future__ import annotations
from typing import Callable
from uuid import UUID
from fastapi import FastAPI, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from .database import Repository
from .delivery import DeliveryBinding, ServerReshare
from .errors import PublicError, ErrorCode

class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)

class ReceiverKey(StrictModel):
    public_key_hex: str=Field(min_length=64,max_length=64)
    expected_previous_version: int | None=Field(default=None,ge=0)
class NewDocument(StrictModel):
    organization_id: str=Field(min_length=1,max_length=128)
    internal_title: str=Field(max_length=1024)
class NewVersion(StrictModel):
    block_count: int=Field(gt=0,le=2147483647)
    model_numeric_version: str=Field(pattern='^[0-9a-f]{64}$')
class NewPreparation(StrictModel):
    block_index: int=Field(ge=0,le=2147483647)
    session_id: str | None=Field(default=None,pattern='^[0-9a-f-]{36}$')
class ACLChange(StrictModel):
    user_id: str=Field(pattern='^[0-9a-f-]{36}$')
    allowed: bool
class DeliveryStage(StrictModel):
    binding: dict
    preparation_id: str=Field(pattern='^[0-9a-f-]{36}$')
    reshare_hex: str=Field(min_length=24592,max_length=24592)
    package_hex: str=Field(min_length=24688,max_length=24688)
class AcceptQuery(StrictModel):
    manifest_sha256: str=Field(pattern='^[0-9a-f]{64}$')
    model_numeric_version: str=Field(pattern='^[0-9a-f]{64}$')
    session_id: str | None=Field(default=None,pattern='^[0-9a-f-]{36}$')


def create_app(repository: Repository, *, activation_check: Callable[[], None] | None=None) -> FastAPI:
    app=FastAPI(title='Private RAG management v1',docs_url=None,redoc_url=None,openapi_url=None)

    @app.exception_handler(PublicError)
    async def public_error(request,exc):
        status={ErrorCode.AUTHENTICATION_FAILED:401,ErrorCode.NOT_AUTHORIZED:403,ErrorCode.BUSY:409,ErrorCode.BACKEND_NOT_READY:503}.get(exc.code,400)
        return JSONResponse({'error':exc.code.value},status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request,exc):
        # FastAPI's default response echoes submitted values. Never use it here.
        return JSONResponse({'error':ErrorCode.BAD_FORMAT.value},status_code=400)

    @app.exception_handler(Exception)
    async def internal_error(request,exc):
        return JSONResponse({'error':ErrorCode.BAD_STATE.value},status_code=500)

    def identity(request: Request):
        peer=request.scope.get('tls_peer')
        if not isinstance(peer,dict) or not isinstance(peer.get('certificate_der'),bytes) or not peer['certificate_der']:
            raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
        import hashlib
        fingerprint=hashlib.sha256(peer['certificate_der']).hexdigest()
        if peer.get('fingerprint_sha256')!=fingerprint: raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
        return repository.authenticate(fingerprint)

    def provider(a=Depends(identity)):
        if a['role']!='D': raise PublicError(ErrorCode.NOT_AUTHORIZED)
        return a['account_id']
    def user(a=Depends(identity)):
        if a['role']!='U': raise PublicError(ErrorCode.NOT_AUTHORIZED)
        return a['account_id']

    def parsehex(value):
        try: return bytes.fromhex(value)
        except ValueError: raise PublicError(ErrorCode.BAD_FORMAT) from None

    @app.get('/v1/status')
    def status(a=Depends(identity)):
        return {'format_version':1,'management':'ready','private_computation':'separately_gated'}

    @app.get('/v1/receiver-keys')
    def key_state(u=Depends(user)):
        return repository.receiver_key_state(u)

    @app.post('/v1/receiver-keys')
    def keys(body:ReceiverKey,u=Depends(user)):
        return {'key_version':repository.register_receiver_key(u,parsehex(body.public_key_hex),body.expected_previous_version)}

    @app.post('/v1/documents')
    def documents(body:NewDocument,d=Depends(provider)):
        return {'document_id':repository.create_document(d,body.organization_id,body.internal_title)}

    @app.post('/v1/documents/{document}/versions')
    def versions(document:UUID,body:NewVersion,d=Depends(provider)):
        return {'document_version':repository.create_version(d,str(document),body.block_count,body.model_numeric_version)}

    @app.put('/v1/documents/{document}/authorization')
    def authorize(document:UUID,body:ACLChange,d=Depends(provider)):
        return {'authorization_generation':repository.set_authorization(d,str(document),body.user_id,body.allowed)}

    @app.post('/v1/documents/{document}/versions/{version}/preparations')
    def prepare(document:UUID,version:int,body:NewPreparation,d=Depends(provider)):
        if activation_check is None: raise PublicError(ErrorCode.BACKEND_NOT_READY)
        session=repository.start_preparation(d,str(document),version,body.block_index,activation_check=activation_check,session_id=body.session_id)
        # No input Tokens are accepted by this management endpoint.
        return repository.preparation_descriptor(d,session)

    @app.get('/v1/documents/{document}/versions/{version}/delivery-plan/{recipient}')
    def plan(document:UUID,version:int,recipient:UUID,d=Depends(provider)):
        return repository.delivery_plan(d,str(document),version,str(recipient))

    @app.post('/v1/documents/{document}/deliveries')
    def stage(document:UUID,body:DeliveryStage,d=Depends(provider)):
        try: binding=DeliveryBinding(**body.binding)
        except TypeError: raise PublicError(ErrorCode.BAD_FORMAT) from None
        message=ServerReshare.decode(parsehex(body.reshare_hex),body.preparation_id)
        return {'delivery_id':repository.stage_delivery(d,str(document),binding,message,parsehex(body.package_hex))}

    @app.post('/v1/documents/{document}/versions/{version}/publish')
    def publish(document:UUID,version:int,d=Depends(provider)):
        repository.publish_version(d,str(document),version)
        return {'status':'published','document_version':version}

    @app.get('/v1/sync')
    def manifest(u=Depends(user)):
        return repository.sync_manifest(u)

    @app.get('/v1/sync/packages/{sha256}')
    def package(sha256:str,u=Depends(user)):
        return Response(repository.sync_package(u,sha256),media_type='application/octet-stream')

    @app.post('/v1/queries/accept')
    def accept(body:AcceptQuery,u=Depends(user)):
        # Eligibility, complete snapshot and cryptographic activation are
        # checked together under one transaction; no preliminary-N race.
        return repository.accept_query(u,body.manifest_sha256,body.model_numeric_version,activation_check=activation_check,session_id=body.session_id)

    @app.get('/v1/sessions/{session}')
    def sessions(session:UUID,a=Depends(identity)):
        return repository.session_status(a['account_id'],str(session))

    @app.post('/v1/sessions/{session}/cancel')
    def cancel(session:UUID,a=Depends(identity)):
        # There is deliberately no reason/status field accepting a secret bit.
        return repository.request_session_cancel(a['account_id'],str(session))

    return app
