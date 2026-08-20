import zipfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import memora.api.main as api_main
from memora.projects import ProjectCatalog


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (40, 30), (80, 130, 170)).save(output, format="JPEG")
    return output.getvalue()


def test_project_catalog_isolated_paths_and_traversal_protection(tmp_path):
    catalog = ProjectCatalog(tmp_path / "projects")
    first = catalog.create("Summer trip")
    second = catalog.create("Family")

    assert first.id != second.id
    assert catalog.index_path(first.id) != catalog.index_path(second.id)
    assert [project.name for project in catalog.list()] == ["Family", "Summer trip"]
    with pytest.raises(ValueError, match="Unsafe upload path"):
        catalog.safe_upload_path(first.id, "../outside.jpg")


def test_project_upload_analyze_and_export_api(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "projects", ProjectCatalog(tmp_path / "projects"))
    client = TestClient(api_main.app)

    created = client.post("/projects", json={"name": "Test album"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    uploaded = client.post(
        f"/projects/{project_id}/files",
        files=[("files", ("photo.jpg", _jpeg(), "image/jpeg"))],
        data={"relative_paths": "day-one/photo.jpg"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["project"]["photo_count"] == 1

    analyzed = client.post(f"/projects/{project_id}/analyze")
    assert analyzed.status_code == 200
    assert analyzed.json()["project"]["status"] == "ready"

    photos = client.get(f"/projects/{project_id}/photos").json()["photos"]
    assert photos[0]["relative_path"] == "day-one/photo.jpg"
    assert client.get(photos[0]["url"]).status_code == 200

    manifest = client.get(f"/projects/{project_id}/export/manifest")
    csv_response = client.get(f"/projects/{project_id}/export/photos.csv")
    archive = client.get(f"/projects/{project_id}/export/best-shots.zip")
    assert manifest.json()["format"] == "memora-project-v1"
    assert "photo_id,filename" in csv_response.text
    with zipfile.ZipFile(BytesIO(archive.content)) as output:
        assert "best-shots.json" in output.namelist()
        assert any(name.startswith("photos/") for name in output.namelist())
