"""SQLite vector search helpers using sqlite-vec extension."""
import sqlite3
import struct


def load_extension(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec extension. Returns True on success."""
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        return True
    except Exception:
        return False


def serialize_vector(vec: list[float]) -> bytes:
    """Convert float list to bytes for sqlite-vec."""
    return struct.pack(f'{len(vec)}f', *vec)


def deserialize_vector(data: bytes, dim: int) -> list[float]:
    """Convert bytes back to float list."""
    return list(struct.unpack(f'{dim}f', data))
