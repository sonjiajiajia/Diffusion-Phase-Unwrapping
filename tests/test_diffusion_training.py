import types
import unittest
from unittest.mock import patch

import torch

from improved_diffusion import train_util
from improved_diffusion.script_util import create_gaussian_diffusion, model_and_diffusion_defaults


class DiffusionTrainingTests(unittest.TestCase):
    def test_gradient_clipping_records_unclipped_norm(self):
        loop = train_util.TrainLoop.__new__(train_util.TrainLoop)
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3., 4.])
        loop.master_params = [parameter]
        loop.max_grad_norm = 1.0
        with patch.object(train_util.logger, "logkv_mean") as log:
            loop._log_grad_norm()
        log.assert_called_once_with("grad_norm", 5.0)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.6, 0.8]))

    def test_nonfinite_gradient_does_not_update_weights(self):
        loop = train_util.TrainLoop.__new__(train_util.TrainLoop)
        parameter = torch.nn.Parameter(torch.tensor(1.))
        parameter.grad = torch.tensor(float("nan"))
        loop.master_params = [parameter]
        loop.max_grad_norm = 1.0
        loop.opt = torch.optim.AdamW([parameter], lr=1e-4)
        with self.assertRaises(RuntimeError):
            loop.optimize_normal()
        self.assertEqual(parameter.item(), 1.0)
        self.assertFalse(loop.opt.state)

    def test_fp16_persistent_overflow_stops_at_unit_scale(self):
        loop = train_util.TrainLoop.__new__(train_util.TrainLoop)
        parameter = torch.nn.Parameter(torch.tensor(1.))
        parameter.grad = torch.tensor(float("inf"))
        loop.model_params = [parameter]
        loop.lg_loss_scale = 1.0
        with self.assertRaises(FloatingPointError):
            loop.optimize_fp16()

    def test_default_predicts_xstart(self):
        self.assertTrue(model_and_diffusion_defaults()["predict_xstart"])
        self.assertEqual(create_gaussian_diffusion().model_mean_type.name, "START_X")

    def test_mse_targets_and_gradients(self):
        image = torch.linspace(-0.8, 0.8, 32).reshape(2, 1, 4, 4)
        noise = torch.linspace(-1, 1, 32).reshape_as(image)
        for predict_xstart in (False, True):
            with self.subTest(predict_xstart=predict_xstart):
                diffusion = create_gaussian_diffusion(predict_xstart=predict_xstart)
                target = image if predict_xstart else noise
                error = torch.nn.Parameter(torch.tensor(0.25))
                result = diffusion.training_losses(
                    lambda x, t, cond: target + error, image, torch.tensor([0, 999]),
                    model_kwargs={"cond": image}, noise=noise,
                )
                torch.testing.assert_close(result["loss"], torch.full((2,), 0.0625))
                result["loss"].mean().backward()
                torch.testing.assert_close(error.grad, torch.tensor(0.5))

    def test_mask_excludes_invalid_pixels(self):
        diffusion = create_gaussian_diffusion(predict_xstart=True)
        image = torch.zeros(2, 1, 2, 2)
        prediction = torch.tensor([[[[1., 100.], [3., 100.]]], [[[7., 7.], [7., 7.]]]], requires_grad=True)
        mask = torch.tensor([[[[1., 0.], [1., 0.]]], [[[0., 0.], [0., 0.]]]])
        result = diffusion.training_losses(
            lambda x, t, cond: prediction, image, torch.tensor([0, 500]),
            model_kwargs={"cond": image, "loss_mask": mask}, noise=torch.zeros_like(image),
        )
        torch.testing.assert_close(result["loss"], torch.tensor([5., 0.]))
        result["loss"].sum().backward()
        torch.testing.assert_close(prediction.grad[mask == 0], torch.zeros(6))

    def test_kl_path_returns_prediction(self):
        diffusion = create_gaussian_diffusion(use_kl=True, predict_xstart=True)
        image = torch.zeros(1, 1, 2, 2)
        result = diffusion.training_losses(lambda x, t: torch.zeros_like(x), image, torch.tensor([1]))
        self.assertTrue(torch.isfinite(result["loss"]).all())
        self.assertEqual(result["pred_xstart"].shape, image.shape)

    def test_microbatch_gradients_match_full_batch(self):
        class FixedSampler:
            def sample(self, size, device):
                return torch.zeros(size, dtype=torch.long), torch.ones(size)

        class QuadraticDiffusion:
            def training_losses(self, model, image, t, model_kwargs):
                return {"loss": (model.parameter - image.flatten(1).mean(1)).square()}

        for microbatch in (8, 2, 3):
            loop = train_util.TrainLoop.__new__(train_util.TrainLoop)
            model = types.SimpleNamespace(parameter=torch.nn.Parameter(torch.tensor(1.)))
            loop.model_params = [model.parameter]
            loop.ddp_model = model
            loop.microbatch = microbatch
            loop.use_ddp = loop.use_fp16 = False
            loop.diffusion = QuadraticDiffusion()
            loop.schedule_sampler = FixedSampler()
            image = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1)
            with patch.object(train_util.dist_util, "dev", return_value=torch.device("cpu")), patch.object(train_util, "log_loss_dict"):
                loop.forward_backward({"image": image, "cond": image})
            torch.testing.assert_close(model.parameter.grad, torch.tensor(-5.))


if __name__ == "__main__":
    unittest.main()
