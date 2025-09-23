from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class AICoCreatorException(Exception):
    """Base exception for AI Co-Creator application."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)


class VideoNotFoundException(AICoCreatorException):
    """Raised when a video is not found."""

    def __init__(self, video_id: int):
        super().__init__(
            message=f"Video with ID {video_id} not found",
            error_code="VIDEO_NOT_FOUND",
            context={"video_id": video_id},
        )


class VideoProcessingException(AICoCreatorException):
    """Raised when video processing fails."""

    def __init__(self, message: str, video_id: Optional[int] = None):
        super().__init__(
            message=f"Video processing failed: {message}",
            error_code="VIDEO_PROCESSING_ERROR",
            context={"video_id": video_id} if video_id else {},
        )


class ModelLoadException(AICoCreatorException):
    """Raised when AI model loading fails."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model {model_name}: {reason}",
            error_code="MODEL_LOAD_ERROR",
            context={"model_name": model_name, "reason": reason},
        )


class InsufficientResourcesException(AICoCreatorException):
    """Raised when system resources are insufficient."""

    def __init__(self, resource_type: str, required: str, available: str):
        super().__init__(
            message=f"Insufficient {resource_type}: required {required}, available {available}",
            error_code="INSUFFICIENT_RESOURCES",
            context={
                "resource_type": resource_type,
                "required": required,
                "available": available,
            },
        )


class FileUploadException(AICoCreatorException):
    """Raised when file upload fails."""

    def __init__(self, reason: str, filename: Optional[str] = None):
        super().__init__(
            message=f"File upload failed: {reason}",
            error_code="FILE_UPLOAD_ERROR",
            context={"filename": filename} if filename else {},
        )


class ValidationException(AICoCreatorException):
    """Raised when validation fails."""

    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Validation error for {field}: {message}",
            error_code="VALIDATION_ERROR",
            context={"field": field},
        )


# HTTP Exception helpers
def to_http_exception(exc: AICoCreatorException) -> HTTPException:
    """Convert custom exception to HTTP exception."""

    error_mapping = {
        "VIDEO_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "VIDEO_PROCESSING_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "MODEL_LOAD_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "INSUFFICIENT_RESOURCES": status.HTTP_507_INSUFFICIENT_STORAGE,
        "FILE_UPLOAD_ERROR": status.HTTP_400_BAD_REQUEST,
        "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
    }

    status_code = error_mapping.get(
        exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    detail = {
        "message": exc.message,
        "error_code": exc.error_code,
        "context": exc.context,
    }

    return HTTPException(status_code=status_code, detail=detail)
