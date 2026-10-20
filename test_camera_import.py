"""CPU control checks must not initialize the model package or CUDA."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class CameraImportTests(unittest.TestCase):
    def run_cpu(self, code: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parent,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_camera_import_does_not_load_model_package(self) -> None:
        self.run_cpu(
            "import cam_controller, sys; "
            "assert 'wan' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )

    def test_control_reduction_runs_without_model_or_cuda(self) -> None:
        self.run_cpu(
            "import numpy as np, sys; "
            "from interactive_runtime import CameraState, InteractiveRuntime; "
            "rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3))); "
            "rt.accept({'forward': 1.0}); "
            "candidate = rt.begin_chunk()['candidate_camera']; "
            "assert np.isfinite(candidate.pose).all(); "
            "assert 'wan' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )


if __name__ == "__main__":
    unittest.main()
