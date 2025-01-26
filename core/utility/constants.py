from enum import Enum


class APIScope(Enum):
    ESIGNATURE = "signature"
    NAVIGATOR = "adm_store_unified_repo_read"

class Versions(Enum):
    LATEST = -1

class SignStatus(Enum):
    REVIEW = "REVIEW",
    SIGN = "SIGN",
    FINALISED = "FINALISED"

