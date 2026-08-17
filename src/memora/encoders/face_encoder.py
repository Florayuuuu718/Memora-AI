"""Optional InsightFace adapter kept separate from clustering algorithms."""

import numpy as np


class InsightFaceEncoder:
    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0) -> None:
        from insightface.app import FaceAnalysis  # type: ignore

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id)

    def detect(self, image: np.ndarray) -> list[dict]:
        faces = self.app.get(image)
        return [
            {
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding.tolist(),
                "det_score": float(face.det_score),
            }
            for face in faces
        ]

