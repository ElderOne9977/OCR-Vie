"""Bound request bodies before Starlette's multipart parser can spool them."""
from contextvars import ContextVar
from uuid import uuid4

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

from app.core.config import get_settings

request_id = ContextVar('ocr_request_id', default='')


class RequestBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        identifier = uuid4().hex
        token = request_id.set(identifier)
        # Bound multipart metadata as well as the image; per-file cap remains in API.
        limit = get_settings().max_upload_bytes + 64 * 1024
        received = 0
        exceeded = False
        replied = False

        async def traced_send(message):
            if message['type'] == 'http.response.start':
                message = {**message, 'headers': [*message.get('headers', []),
                            (b'x-request-id', identifier.encode())]}
            await send(message)

        async def reject():
            nonlocal replied
            replied = True
            response = JSONResponse({'detail': {'code': 'FILE_TOO_LARGE', 'message': 'Request body too large'}},
                                    status_code=413, headers={'Connection': 'close'})
            await response(scope, receive, traced_send)

        async def bounded_receive():
            nonlocal received, exceeded
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > limit:
                    exceeded = True
                    # Starlette closes already-spooled files for MultiPartException.
                    raise MultiPartException('Request body too large')
            return message

        async def bounded_send(message):
            if exceeded:
                if not replied:
                    await reject()
                return
            await traced_send(message)

        try:
            lengths = [v for k, v in scope.get('headers', []) if k.lower() == b'content-length']
            if lengths and (len(lengths) != 1 or not lengths[0].isdigit()):
                await JSONResponse({'detail': 'Invalid Content-Length'}, status_code=400)(scope, receive, traced_send)
            elif lengths and int(lengths[0]) > limit:
                await reject()
            else:
                try:
                    await self.app(scope, bounded_receive, bounded_send)
                except MultiPartException:
                    if not exceeded:
                        raise
                    if not replied:
                        await reject()
        finally:
            request_id.reset(token)
