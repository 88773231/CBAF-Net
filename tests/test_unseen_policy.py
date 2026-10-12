import unittest

import torch

from analysis.screen_unseen_policy import fit_fold_transforms, inner_profile_partition


class UnseenPolicyTests(unittest.TestCase):
    def test_heldout_policy_never_enters_fitting_or_selection(self):
        rows, policies = [], {}
        for index in range(24):
            profile = f"profile{index}"
            policies[profile] = "excluded" if index < 6 else "remaining"
            rows.extend({"base_profile_id": profile, "label": label} for label in range(3))
        fitting, selection = inner_profile_partition(rows, policies, "excluded", 42)
        self.assertFalse(set(fitting) & set(selection))
        for index in [*fitting, *selection]:
            self.assertEqual(policies[rows[index]["base_profile_id"]], "remaining")
        self.assertEqual(len(fitting), 15 * 3)
        self.assertEqual(len(selection), 3 * 3)

    def test_fold_standardizers_ignore_challenge_and_inner_selection(self):
        def values(value):
            return {"raw_tabular": torch.full((4, 24), value),
                    "action": torch.zeros(4, 5, 15)}
        fitting, selection, challenge = values(1), values(100), values(-100)
        transform = fit_fold_transforms(
            fitting, selection, challenge,
            (torch.ones(4, 5, 7), torch.full((4, 5, 7), 100.0),
             torch.full((4, 5, 7), -100.0)),
        )
        self.assertEqual(transform["table_mean"], [1.0] * 24)
        self.assertEqual(transform["step_mean"], [1.0] * 7)
        torch.testing.assert_close(fitting["tabular"], torch.zeros(4, 24))
        torch.testing.assert_close(challenge["action"][:, :, -7:], torch.zeros(4, 5, 7))


if __name__ == "__main__":
    unittest.main()
