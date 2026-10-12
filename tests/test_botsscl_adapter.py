import unittest

import torch

from src.botsscl_adapter import corrupt_features, nt_xent


class BotSSCLAdapterTests(unittest.TestCase):
    def test_exact_corruption_fraction_and_boundaries(self):
        anchor, donor = torch.zeros(4, 10), torch.ones(4, 10)
        result = corrupt_features(anchor, donor, 0.6, torch.Generator().manual_seed(42))
        torch.testing.assert_close(result.sum(1), torch.full((4,), 6.0))
        torch.testing.assert_close(
            corrupt_features(anchor, donor, 0, torch.Generator()), anchor
        )
        torch.testing.assert_close(
            corrupt_features(anchor, donor, 1, torch.Generator()), donor
        )

    def test_cpu_loss_finite_symmetric_and_differentiable(self):
        left, right = torch.randn(8, 16, requires_grad=True), torch.randn(8, 16)
        loss = nt_xent(left, right)
        torch.testing.assert_close(loss, nt_xent(right, left))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(left.grad.abs().sum()), 0)

    def test_invalid_small_batches_rejected(self):
        with self.assertRaises(ValueError):
            nt_xent(torch.randn(1, 4), torch.randn(1, 4))


if __name__ == "__main__":
    unittest.main()
