import copy
import unittest

import torch

from src.botdmm_development import DevelopmentGraphView, forward_seed_logits, make_backbone
from src.botdmm_module_controls import make_module_control, train_module_control
from src.data_loader import StrictOneHopNeighborCollator
from tests import test_botdmm_development as fixture_module


class BotDMMModuleControlTests(unittest.TestCase):
    def fixture(self):
        helpers = fixture_module.BotDMMDevelopmentTests()
        values, graph = helpers.values()
        view = DevelopmentGraphView(values, graph, "botdmm_timing", 2)
        batch = StrictOneHopNeighborCollator(view, expected_k=2)(
            [{"_index": index} for index in range(12)]
        )
        return view, batch, helpers.config()

    def test_module_removals_preserve_same_seed_parameter_initialization(self):
        view, _, config = self.fixture()
        torch.manual_seed(41)
        baseline = make_backbone(view, 3, config)
        for method in ("botdmm_without_structure", "botdmm_without_orthogonal"):
            torch.manual_seed(41)
            candidate = make_module_control(view, 3, config, method)
            self.assertEqual(set(baseline.state_dict()), set(candidate.state_dict()))
            for name in baseline.state_dict():
                torch.testing.assert_close(baseline.state_dict()[name], candidate.state_dict()[name])
        with self.assertRaises(ValueError):
            make_module_control(view, 3, config, "unknown")

    def test_structure_mask_blocks_branch_classification_gradients(self):
        view, batch, config = self.fixture()
        model = make_module_control(view, 3, config, "botdmm_without_structure")
        torch.nn.functional.cross_entropy(forward_seed_logits(model, batch),
                                          batch["seed_labels"]).backward()
        self.assertTrue(all(parameter.grad is None
                            for parameter in model.structure_decoupler.parameters()))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in model.content_decoupler.parameters()))

    def test_no_orthogonal_matches_alpha_zero_and_structure_mask_makes_it_redundant(self):
        view, batch, config = self.fixture()
        baseline = make_backbone(view, 3, config).eval()
        zero_alpha = copy.deepcopy(baseline)
        zero_alpha.orthogonal_regularizer.alpha.fill_(0)
        removed = copy.deepcopy(baseline)
        removed.ablation_mode = "no_orthogonal"
        with torch.no_grad():
            torch.testing.assert_close(
                forward_seed_logits(zero_alpha, batch), forward_seed_logits(removed, batch),
                atol=1e-6, rtol=0,
            )
            structure_mask = copy.deepcopy(baseline)
            structure_mask.ablation_mode = "no_structure"
            both = copy.deepcopy(structure_mask)
            both.orthogonal_regularizer.alpha.fill_(0)
            torch.testing.assert_close(
                forward_seed_logits(structure_mask, batch), forward_seed_logits(both, batch),
                atol=1e-6, rtol=0,
            )

    def test_one_epoch_both_controls_finite_and_disjoint(self):
        helpers = fixture_module.BotDMMDevelopmentTests()
        fitting_values, fitting_graph = helpers.values("f")
        select_values, select_graph = helpers.values("s")
        fitting = DevelopmentGraphView(fitting_values, fitting_graph, "botdmm_timing", 2)
        selection = DevelopmentGraphView(select_values, select_graph, "botdmm_timing", 2)
        policies = {f"{prefix}{index}": "a" if index < 6 else "b"
                    for prefix in ("f", "s") for index in range(12)}
        for method in ("botdmm_without_structure", "botdmm_without_orthogonal"):
            _, fitted = train_module_control(
                fitting, selection, select_values, policies, 3, 42, helpers.config(), method,
            )
            self.assertTrue(torch.isfinite(torch.tensor(fitted["history"][0]["training_loss"])))
            self.assertTrue(fitted["parameter_layout_preserved"])
        with self.assertRaises(ValueError):
            train_module_control(fitting, fitting, fitting_values, policies, 3, 42,
                                 helpers.config(), "botdmm_without_structure")


if __name__ == "__main__":
    unittest.main()
