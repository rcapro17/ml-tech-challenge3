from __future__ import annotations

import os
from typing import Optional
import fsspec

def clean_env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return None
    # remove espaços em volta e quebras de linha acidentais
    return v.strip().replace("\r", "").replace("\n", "")

def get_fs_for_uri(uri: str | None):
    """
    Retorna um filesystem do fsspec compatível com a URI.
    - s3://... -> S3FileSystem (com credenciais limpas)
    - caminho local -> LocalFileSystem
    """
    if not uri or "://" not in uri:
        # local
        return fsspec.filesystem("file")

    scheme = uri.split("://", 1)[0].lower()
    if scheme == "s3":
        key = clean_env("AWS_ACCESS_KEY_ID")
        secret = clean_env("AWS_SECRET_ACCESS_KEY")
        token = clean_env("AWS_SESSION_TOKEN")  # opcional
        region = clean_env("AWS_DEFAULT_REGION")
        client_kwargs = {}
        if region:
            client_kwargs["region_name"] = region
        # não usar endpoint_url a menos que necessário
        fs = fsspec.filesystem(
            "s3",
            key=key,
            secret=secret,
            token=token,
            anon=False,
            client_kwargs=client_kwargs,
            default_cache_type="none",
        )
        return fs
    # fallback por segurança
    return fsspec.filesystem("file")

def join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/").replace("\\", "/") for p in parts if p)
    return f"{base}/{suffix}" if suffix else base
