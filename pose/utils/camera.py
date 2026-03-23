import json
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsic:
    cx: float
    cy: float
    fx: float
    fy: float
    depth_scale: float
    height: int
    width: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)

        return cls(
            cx=data["cx"],
            cy=data["cy"],
            fx=data["fx"],
            fy=data["fy"],
            depth_scale=data["depth_scale"],
            height=data["height"],
            width=data["width"],
        )


__all__ = ["CameraIntrinsic"]
