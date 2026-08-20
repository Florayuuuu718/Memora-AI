from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


@dataclass
class PhotoProject:
    id: str
    name: str
    created_at: str
    updated_at: str
    photo_count: int = 0
    analyzed_count: int = 0
    status: str = "empty"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectCatalog:
    """Filesystem-backed project catalog with one private index per photo folder."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _validate_id(self, project_id: str) -> None:
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ValueError("Invalid project ID")

    def directory(self, project_id: str) -> Path:
        self._validate_id(project_id)
        return self.root / project_id

    def uploads(self, project_id: str) -> Path:
        return self.directory(project_id) / "uploads"

    def index_path(self, project_id: str) -> Path:
        return self.directory(project_id) / "index.json"

    def people_path(self, project_id: str) -> Path:
        return self.directory(project_id) / "people.json"

    def annotations_path(self, project_id: str) -> Path:
        return self.directory(project_id) / "annotations.json"

    def load_annotations(self, project_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        path = self.annotations_path(project_id)
        if not path.exists():
            return {"people": {}, "events": {}, "journeys": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"people": {}, "events": {}, "journeys": {}}
        return {
            "people": dict(value.get("people", {})),
            "events": dict(value.get("events", {})),
            "journeys": dict(value.get("journeys", {})),
        }

    def save_annotations(self, project_id: str, value: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.annotations_path(project_id).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _metadata_path(self, project_id: str) -> Path:
        return self.directory(project_id) / "project.json"

    def create(self, name: str) -> PhotoProject:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name must not be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        project_id = uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        project = PhotoProject(project_id, clean_name, now, now)
        self.uploads(project_id).mkdir(parents=True)
        self.save(project)
        return project

    def save(self, project: PhotoProject) -> None:
        directory = self.directory(project.id)
        directory.mkdir(parents=True, exist_ok=True)
        self._metadata_path(project.id).write_text(
            json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, project_id: str) -> PhotoProject:
        path = self._metadata_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return PhotoProject(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[PhotoProject]:
        if not self.root.exists():
            return []
        projects: list[PhotoProject] = []
        for path in self.root.glob("*/project.json"):
            try:
                projects.append(PhotoProject(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda value: value.updated_at, reverse=True)

    def touch(self, project_id: str, **changes: Any) -> PhotoProject:
        project = self.get(project_id)
        for key, value in changes.items():
            if not hasattr(project, key):
                raise ValueError(f"Unknown project field: {key}")
            setattr(project, key, value)
        project.updated_at = datetime.now(timezone.utc).isoformat()
        self.save(project)
        return project

    def safe_upload_path(self, project_id: str, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Unsafe upload path: {relative_path}")
        clean_parts = [part for part in normalized.parts if part not in {"", "."}]
        if not clean_parts:
            raise ValueError("Upload filename must not be empty")
        destination = self.uploads(project_id).joinpath(*clean_parts)
        root = self.uploads(project_id).resolve()
        if root not in destination.resolve().parents:
            raise ValueError(f"Unsafe upload path: {relative_path}")
        return destination
