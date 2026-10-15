"""Registro local y optativo de peticiones HTTP con ID de sesión."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from pathlib import Path


def safe_headers(headers) -> list[list[str]]:
    """Devuelve las cabeceras de entrada/salida, ocultando credenciales; headers conserva duplicados."""
    sensitive = ("auth", "cookie", "token", "secret", "password", "key", "credential", "signature")
    return [[name, "[REDACTED]" if any(part in name.lower() for part in sensitive) else value]
            for name, value in headers]


def body_text(body: bytes) -> dict:
    """Representa los bytes íntegros como UTF-8 o, si no es posible, como base64."""
    try:
        return {"body": body.decode("utf-8"), "body_encoding": "utf-8"}
    except UnicodeDecodeError:
        import base64
        return {"body": base64.b64encode(body).decode("ascii"), "body_encoding": "base64"}


class SessionRequestLog:
    """Conserva las últimas limit peticiones identificadas en JSONL, incluso tras reinicios."""

    def __init__(self, path: str | Path, limit: int = 20):
        """Inicializa ruta y límite; rechaza rutas inseguras y límites no positivos."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("session request log limit must be a positive integer")
        self.path = Path(path).absolute()
        if not self.path.name or not self.path.parent.is_dir():
            raise ValueError("session request log needs an existing parent directory")
        self.limit = limit
        self.lock = threading.Lock()
        self._check_path()

    def _check_path(self) -> None:
        """Impide reemplazar enlaces o archivos ajenos que no sean un log privado."""
        try:
            mode = self.path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise ValueError(f"session request log must be a private regular file (0600): {self.path}")

    def append(self, entry: dict) -> None:
        """Añade una entrada y reemplaza atómicamente el archivo, reteniendo solo limit entradas."""
        with self.lock:
            self._check_path()
            previous = []
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as source:
                    for line in source:
                        item = json.loads(line)
                        if not isinstance(item, dict) or item.get("type") != "session_request":
                            raise ValueError(f"not a session request log: {self.path}")
                        previous.append(item)
                        if len(previous) > self.limit - 1:
                            previous.pop(0)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".session-request-", delete=False) as output:
                    temporary = output.name
                    for item in (*previous, entry):
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                self._check_path()
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
