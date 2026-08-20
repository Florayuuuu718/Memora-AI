import json
from io import BytesIO

import numpy as np
from PIL import Image

from memora.integrations.immich import ImmichClient, sync_immich_assets
from memora.models import PhotoRecord
from memora.retrieval.vector_store import NumpyVectorStore


class _Response:
    def __init__(self, payload: bytes, content_type: str = "application/json") -> None:
        self.payload = payload
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return self.payload


class _Encoder:
    def encode_image(self, _path):
        return np.asarray([0.6, 0.8], dtype=np.float32)


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (80, 120, 160)).save(output, format="JPEG")
    return output.getvalue()


def test_client_normalizes_url_authenticates_and_paginates():
    requests = []
    pages = [
        {"assets": {"items": [{"id": "asset-1"}], "nextPage": "2"}},
        {"assets": {"items": [{"id": "asset-2"}], "nextPage": None}},
    ]

    def opener(request, timeout):
        requests.append((request, timeout))
        body = json.loads(request.data)
        return _Response(json.dumps(pages[int(body["page"]) - 1]).encode())

    client = ImmichClient("https://photos.example.test/", "secret", opener=opener)
    assets = client.list_image_assets(page_size=25)

    assert [asset["id"] for asset in assets] == ["asset-1", "asset-2"]
    assert requests[0][0].full_url == "https://photos.example.test/api/search/metadata"
    assert requests[0][0].headers["X-api-key"] == "secret"
    assert json.loads(requests[0][0].data)["withExif"] is True


def test_sync_indexes_immich_metadata_and_reuses_unchanged_assets(tmp_path):
    asset = {
        "id": "asset-1",
        "updatedAt": "2026-08-18T10:00:00Z",
        "fileCreatedAt": "2025-05-01T09:30:00Z",
        "originalFileName": "beach.jpg",
        "width": 4000,
        "height": 3000,
        "exifInfo": {"latitude": 22.3, "longitude": 114.2, "make": "Fuji", "model": "X100"},
    }

    class Client:
        thumbnail_calls = 0

        def list_image_assets(self, *, page_size):
            assert page_size == 50
            return [asset]

        def thumbnail(self, asset_id, *, size):
            assert (asset_id, size) == ("asset-1", "preview")
            self.thumbnail_calls += 1
            return _jpeg(), "image/jpeg"

    client = Client()
    store = NumpyVectorStore()
    index_path = tmp_path / "index.json"
    cache_path = tmp_path / "cache"

    first = sync_immich_assets(client, _Encoder(), store, index_path, cache_path, page_size=50)
    second = sync_immich_assets(client, _Encoder(), store, index_path, cache_path, page_size=50)

    assert first.indexed_count == 1
    assert second.reused_count == 1
    assert client.thumbnail_calls == 1
    assert store.records[0].id == "asset-1"
    assert store.records[0].source == "immich"
    assert store.records[0].captured_at_source == "immich"
    assert store.records[0].camera == "Fuji X100"
    assert store.records[0].embedding == [0.6000000238418579, 0.800000011920929]


def test_sync_prunes_only_missing_immich_records(tmp_path):
    class Client:
        def list_image_assets(self, *, page_size):
            return []

    store = NumpyVectorStore(
        [
            PhotoRecord("local", "local.jpg"),
            PhotoRecord("gone", "gone.jpg", source="immich"),
        ]
    )
    result = sync_immich_assets(
        Client(), _Encoder(), store, tmp_path / "index.json", tmp_path / "cache", prune_missing=True
    )

    assert result.removed_count == 1
    assert [record.id for record in store.records] == ["local"]
