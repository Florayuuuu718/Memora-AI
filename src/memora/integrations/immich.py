from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from memora.duplicate.phash import phash
from memora.encoders.clip_encoder import VisionEncoder
from memora.models import PhotoRecord
from memora.quality.best_shot import score_photo
from memora.retrieval.vector_store import NumpyVectorStore


class ImmichError(RuntimeError):
    """A sanitized error returned by the Immich adapter."""


@dataclass
class ImmichSyncResult:
    remote_count: int = 0
    indexed_count: int = 0
    reused_count: int = 0
    removed_count: int = 0
    failed_count: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _api_url(server_url: str) -> str:
    value = server_url.strip().rstrip("/")
    if not value:
        raise ValueError("Immich server URL must not be empty")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Immich server URL must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api"):
        path += "/api"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


class ImmichClient:
    """Small dependency-free client for the stable Immich REST endpoints we use."""

    def __init__(
        self,
        server_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("Immich API key must not be empty")
        self.api_url = _api_url(server_url)
        self.server_url = self.api_url.removesuffix("/api")
        self.api_key = api_key
        self.timeout = timeout
        self._opener = opener

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        url = f"{self.api_url}/{path.lstrip('/')}"
        if query:
            url += "?" + urlencode({key: value for key, value in query.items() if value is not None})
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json", "x-api-key": self.api_key}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return response.read(), response.headers.get("Content-Type", "application/octet-stream")
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                detail = ""
            message = str(detail or exc.reason).replace(self.api_key, "***")
            raise ImmichError(f"Immich returned HTTP {exc.code}: {message}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ImmichError(f"Could not reach Immich: {reason}") from exc

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        payload, _ = self._request(method, path, **kwargs)
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImmichError("Immich returned an invalid JSON response") from exc

    def status(self) -> dict[str, Any]:
        ping = self._json("GET", "/server/ping")
        version = self._json("GET", "/server/version")
        return {"connected": True, "server_url": self.server_url, "ping": ping, "version": version}

    def list_image_assets(self, *, page_size: int = 250) -> list[dict[str, Any]]:
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        assets: list[dict[str, Any]] = []
        page: int | str = 1
        seen_pages: set[str] = set()
        while True:
            response = self._json(
                "POST",
                "/search/metadata",
                body={"type": "IMAGE", "withExif": True, "size": page_size, "page": page},
            )
            page_data = response.get("assets", response) if isinstance(response, dict) else {}
            items = page_data.get("items", []) if isinstance(page_data, dict) else []
            assets.extend(item for item in items if isinstance(item, dict) and item.get("id"))
            next_page = page_data.get("nextPage") if isinstance(page_data, dict) else None
            if next_page is None or str(next_page) in seen_pages:
                break
            seen_pages.add(str(next_page))
            page = next_page
        return assets

    def thumbnail(self, asset_id: str, *, size: str = "preview") -> tuple[bytes, str]:
        return self._request(
            "GET", f"/assets/{quote(asset_id, safe='')}/thumbnail", query={"size": size}
        )

    def create_album(
        self, album_name: str, asset_ids: list[str], *, description: str = ""
    ) -> dict[str, Any]:
        if not album_name.strip():
            raise ValueError("album_name must not be empty")
        response = self._json(
            "POST",
            "/albums",
            body={"albumName": album_name.strip(), "description": description, "assetIds": asset_ids},
        )
        if not isinstance(response, dict):
            raise ImmichError("Immich returned an invalid album response")
        return response


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cache_path(cache_dir: Path, asset_id: str) -> Path:
    safe_id = asset_id if re.fullmatch(r"[A-Za-z0-9_-]+", asset_id) else quote(asset_id, safe="")
    return cache_dir / f"{safe_id}.jpg"


def _record_from_asset(asset: dict[str, Any], path: Path, encoder: VisionEncoder) -> PhotoRecord:
    exif = asset.get("exifInfo") or {}
    camera = " ".join(
        str(value).strip() for value in (exif.get("make"), exif.get("model")) if value
    ) or None
    return PhotoRecord(
        id=str(asset["id"]),
        path=str(path.resolve()),
        width=int(asset.get("width") or exif.get("exifImageWidth") or 0),
        height=int(asset.get("height") or exif.get("exifImageHeight") or 0),
        captured_at=asset.get("fileCreatedAt") or asset.get("localDateTime"),
        captured_at_source="immich",
        latitude=_float(exif.get("latitude")),
        longitude=_float(exif.get("longitude")),
        gps_source="immich" if exif.get("latitude") is not None else None,
        camera=camera,
        embedding=encoder.encode_image(path).tolist(),
        phash=phash(path),
        quality=score_photo(str(path)),
        source="immich",
        source_updated_at=asset.get("updatedAt"),
        original_filename=asset.get("originalFileName"),
    )


def sync_immich_assets(
    client: ImmichClient,
    encoder: VisionEncoder,
    store: NumpyVectorStore,
    index_path: str | Path,
    cache_dir: str | Path,
    *,
    page_size: int = 250,
    force: bool = False,
    prune_missing: bool = False,
) -> ImmichSyncResult:
    """Incrementally mirror Immich image thumbnails into the Memora AI index."""
    assets = client.list_image_assets(page_size=page_size)
    result = ImmichSyncResult(remote_count=len(assets))
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    existing = {record.id: record for record in store.records}
    remote_ids = {str(asset["id"]) for asset in assets}
    synced: list[PhotoRecord] = []

    for asset in assets:
        asset_id = str(asset["id"])
        path = _cache_path(cache, asset_id)
        previous = existing.get(asset_id)
        unchanged = (
            not force
            and previous is not None
            and previous.source == "immich"
            and previous.source_updated_at == asset.get("updatedAt")
            and bool(previous.embedding)
            and path.exists()
        )
        if unchanged:
            previous.path = str(path.resolve())
            synced.append(previous)
            result.reused_count += 1
            continue
        try:
            payload, _ = client.thumbnail(asset_id, size="preview")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
            synced.append(_record_from_asset(asset, path, encoder))
            result.indexed_count += 1
        except (ImmichError, OSError, ValueError) as exc:
            result.failed_count += 1
            result.errors.append({"asset_id": asset_id, "error": str(exc)})
            if previous is not None:
                synced.append(previous)

    other_records = [
        record
        for record in store.records
        if record.source != "immich" and record.id not in remote_ids
    ]
    if not prune_missing:
        synced_ids = {record.id for record in synced}
        synced.extend(
            record
            for record in store.records
            if record.source == "immich" and record.id not in remote_ids and record.id not in synced_ids
        )
    else:
        result.removed_count = sum(
            1 for record in store.records if record.source == "immich" and record.id not in remote_ids
        )
    store.records = other_records + synced
    store.save(index_path)
    return result
