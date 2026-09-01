"""Small S3-compatible transport used by distributed dataset export/loading."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class S3Path:
    """Minimal path-like wrapper around an ``s3://`` object key."""

    bucket: str
    key: str = ""

    @classmethod
    def parse(cls, value: str) -> "S3Path":
        """Parse one S3 URI.

        Returns:
            Parsed S3 path.

        Raises:
            ValueError: If *value* is not a plain ``s3://bucket/key`` URI.
        """
        parsed = urlsplit(value)
        if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"Invalid S3 URI: {value!r}")
        return cls(parsed.netloc, parsed.path.lstrip("/"))

    def __truediv__(self, child: str | PurePosixPath) -> "S3Path":
        child_text = str(child).lstrip("/")
        key = f"{self.key.rstrip('/')}/{child_text}" if self.key else child_text
        return S3Path(self.bucket, key)

    @property
    def suffix(self) -> str:
        """Return the suffix of the object key."""
        return PurePosixPath(self.key).suffix

    def __str__(self) -> str:
        suffix = f"/{self.key}" if self.key else ""
        return f"s3://{self.bucket}{suffix}"


class S3Client:
    """Lazy boto3 wrapper for the few operations NexuML actually needs."""

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        profile: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.region = region
        self.profile = profile
        self._client = client

    def _get_client(self) -> Any:
        """Create the boto3 client on first use.

        Returns:
            A boto3-compatible S3 client.
        """
        if self._client is not None:
            return self._client
        try:
            boto3 = importlib.import_module("boto3")
        except ImportError as error:
            raise RuntimeError("S3 support requires the nexuml[s3] extra") from error

        session = boto3.session.Session(profile_name=self.profile) if self.profile else boto3.Session()
        kwargs: dict[str, Any] = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.region:
            kwargs["region_name"] = self.region
        self._client = session.client("s3", **kwargs)
        return self._client

    def read_bytes(self, uri: str | S3Path) -> bytes:
        """Read one object into memory.

        Returns:
            Object bytes.
        """
        path = uri if isinstance(uri, S3Path) else S3Path.parse(uri)
        response = self._get_client().get_object(Bucket=path.bucket, Key=path.key)
        return response["Body"].read()

    def upload_file(self, source: str | Path, destination: str | S3Path) -> None:
        """Upload one local file to S3."""
        path = destination if isinstance(destination, S3Path) else S3Path.parse(destination)
        self._get_client().upload_file(str(source), path.bucket, path.key)

    def upload_bytes(self, data: bytes, destination: str | S3Path) -> None:
        """Upload one in-memory object to S3."""
        path = destination if isinstance(destination, S3Path) else S3Path.parse(destination)
        self._get_client().put_object(Bucket=path.bucket, Key=path.key, Body=data)

    def download_file(self, source: str | S3Path, destination: str | Path) -> None:
        """Download one object to a local path."""
        path = source if isinstance(source, S3Path) else S3Path.parse(source)
        self._get_client().download_file(path.bucket, path.key, str(destination))

    def list(self, prefix: str | S3Path) -> list[S3Path]:
        """List object paths below an S3 prefix.

        Returns:
            S3 paths in provider order.
        """
        path = prefix if isinstance(prefix, S3Path) else S3Path.parse(prefix)
        paginator = self._get_client().get_paginator("list_objects_v2")
        return [
            S3Path(path.bucket, item["Key"])
            for page in paginator.paginate(Bucket=path.bucket, Prefix=path.key)
            for item in page.get("Contents", [])
            if isinstance(item.get("Key"), str)
        ]


def is_s3_uri(value: object) -> bool:
    """Return whether *value* is an S3 URI."""
    return isinstance(value, str) and value.startswith("s3://")


__all__ = ["S3Client", "S3Path", "is_s3_uri"]
