from enum import Enum


class APIScope(Enum):
    ESIGNATURE = "signature"
    NAVIGATOR = "adm_store_unified_repo_read"

class Versions(Enum):
    LATEST = -1

