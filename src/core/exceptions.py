"""Custom exceptions for Project Atlas."""


class AtlasError(Exception):
    """Base exception for all Project Atlas errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class TenantNotFoundError(AtlasError):
    def __init__(self, tenant_id: str):
        super().__init__(f"Tenant not found: {tenant_id}", "TENANT_NOT_FOUND")


class ProjectNotFoundError(AtlasError):
    def __init__(self, project_id: str):
        super().__init__(f"Project not found: {project_id}", "PROJECT_NOT_FOUND")


class FilingError(AtlasError):
    def __init__(self, message: str):
        super().__init__(message, "FILING_ERROR")


class SearchError(AtlasError):
    def __init__(self, message: str):
        super().__init__(message, "SEARCH_ERROR")


class LLMError(AtlasError):
    def __init__(self, provider: str, message: str):
        super().__init__(f"LLM error ({provider}): {message}", "LLM_ERROR")
        self.provider = provider


class CheckpointError(AtlasError):
    def __init__(self, message: str):
        super().__init__(message, "CHECKPOINT_ERROR")
