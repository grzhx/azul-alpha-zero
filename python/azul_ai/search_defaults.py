"""Current chance-search settings shared by training and both advice frontends."""
CHANCE_CAP = 256
SEARCH_REVISION = 2


def require_current_search(library):
    import ctypes
    revision = getattr(library, "azul_search_revision", None)
    if revision is None:
        raise RuntimeError("Native search is outdated; rebuild Azul to enable factory-equivalent chance merging")
    revision.restype = ctypes.c_int
    actual = revision()
    if actual != SEARCH_REVISION:
        raise RuntimeError(f"Native search revision mismatch: {actual}, expected {SEARCH_REVISION}")
    return actual
