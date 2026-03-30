from src.core.interfaces import BaseAdapter


class AdapterError(RuntimeError):
    pass


__all__ = ["BaseAdapter", "AdapterError"]

