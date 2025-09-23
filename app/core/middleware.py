import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)

from app.core.exceptions import AICoCreatorException, to_http_exception
from app.core.logging_config import LogContext, get_logger

logger = get_logger("middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request
        with LogContext(request_id=request_id):
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                },
            )

            try:
                response = await call_next(request)

                # Calculate duration
                duration = time.time() - start_time

                # Log response
                logger.info(
                    f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                    extra={
                        "status_code": response.status_code,
                        "duration_ms": round(duration * 1000, 2),
                    },
                )

                # Add headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(duration, 4))

                return response

            except Exception as exc:
                duration = time.time() - start_time
                logger.error(
                    f"Request failed: {request.method} {request.url.path}",
                    extra={
                        "duration_ms": round(duration * 1000, 2),
                        "error": str(exc),
                    },
                    exc_info=True,
                )
                raise


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)
        except AICoCreatorException as exc:
            # Convert custom exceptions to HTTP exceptions
            http_exc = to_http_exception(exc)
            logger.warning(
                f"Application exception: {exc.message}",
                extra={
                    "error_code": exc.error_code,
                    "context": exc.context,
                },
            )
            raise http_exc
        except Exception as exc:
            # Log unexpected exceptions
            logger.error(
                f"Unexpected exception: {str(exc)}",
                exc_info=True,
            )
            raise


class CORSHeaderMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers[
            "Access-Control-Allow-Methods"
        ] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
