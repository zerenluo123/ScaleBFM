import smplx
import torch

# AMASS, GRAB and OMOMO store 16 betas; smplx.create builds 10 by default.
NUM_BETAS = 16

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
def run_smplx_inference(body_model, **kwargs):
    # The model's default expression is a single row, which smplx cannot concatenate with per-frame betas.
    kwargs.setdefault("expression", torch.zeros(kwargs["global_orient"].shape[0], body_model.num_expression_coeffs))
    return body_model(**kwargs)
