from types import SimpleNamespace

import smplx
import torch

# AMASS, GRAB and OMOMO store 16 betas; smplx.create builds 10 by default.
NUM_BETAS = 16
# smplx poses the whole mesh of every frame it is given (~1 MB per frame); long AMASS clips at 120-250 fps
# took up to 36 GB in one call.
CHUNK_FRAMES = 1000

class SmplxModelCache:

    def __init__(self, body_model_path):
        self.body_model_path = body_model_path
        self._models = {}

    def get(self, gender):
        gender = str(gender)
        if gender not in self._models:
            self._models[gender] = smplx.create(
                self.body_model_path,
                "smplx",
                gender=gender,
                use_pca=False,
                num_betas=NUM_BETAS,
            )
        return self._models[gender]


@torch.inference_mode()
def run_smplx_inference(body_model, chunk=CHUNK_FRAMES, **kwargs):
    """global_orient, full_pose and joints of every frame, computed chunk frames at a time."""
    num_frames = kwargs["global_orient"].shape[0]
    # The model's default expression is a single row, which smplx cannot concatenate with per-frame betas.
    kwargs.setdefault("expression", torch.zeros(num_frames, body_model.num_expression_coeffs))
    parts = []
    for start in range(0, num_frames, chunk):
        sub = {k: v[start:start + chunk] if torch.is_tensor(v) and v.shape[0] == num_frames and num_frames > 1 else v
               for k, v in kwargs.items()}
        out = body_model(**sub)
        parts.append((out.global_orient, out.full_pose, out.joints))
    global_orient, full_pose, joints = (torch.cat(p) for p in zip(*parts))
    return SimpleNamespace(global_orient=global_orient, full_pose=full_pose, joints=joints)


@torch.inference_mode()
def lowest_vertex_z(body_model, chunk=CHUNK_FRAMES, **kwargs):
    """Lowest z of the posed mesh over all frames, computed chunk frames at a time."""
    num_frames = kwargs["global_orient"].shape[0]
    kwargs.setdefault("expression", torch.zeros(num_frames, body_model.num_expression_coeffs))
    lowest = float("inf")
    for start in range(0, num_frames, chunk):
        sub = {k: v[start:start + chunk] if torch.is_tensor(v) and v.shape[0] == num_frames and num_frames > 1 else v
               for k, v in kwargs.items()}
        lowest = min(lowest, body_model(**sub).vertices[..., 2].min().item())
    return lowest
