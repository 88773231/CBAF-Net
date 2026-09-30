import unittest

import torch

from dataset_pipeline.export_quadbot_v3_features import llm_style_features
from src.data_loader import (
    TABULAR_PREPROCESSING_VERSION,
    apply_column_standardizer,
    fit_column_standardizer,
    tabular_preprocessing_report,
)


class TabularPreprocessingTests(unittest.TestCase):
    def test_training_columns_are_standardized_and_constants_are_zeroed(self):
        train = torch.tensor(
            [
                [1.0, 25.0, 10.0],
                [2.0, 25.0, 20.0],
                [3.0, 25.0, 30.0],
                [4.0, 25.0, 40.0],
            ]
        )
        standardizer = fit_column_standardizer(train)
        transformed = apply_column_standardizer(train, standardizer, clamp=None)

        torch.testing.assert_close(transformed[:, 1], torch.zeros(4))
        torch.testing.assert_close(
            transformed[:, [0, 2]].mean(dim=0),
            torch.zeros(2),
            atol=1e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            transformed[:, [0, 2]].std(dim=0, unbiased=False),
            torch.ones(2),
            atol=1e-6,
            rtol=0.0,
        )

    def test_validation_uses_training_statistics_without_refitting(self):
        train = torch.tensor([[0.0, 7.0], [2.0, 7.0]])
        validation = torch.tensor([[3.0, 99.0], [5.0, -10.0]])
        standardizer = fit_column_standardizer(train)
        transformed = apply_column_standardizer(validation, standardizer, clamp=None)

        torch.testing.assert_close(
            transformed[:, 0],
            torch.tensor([2.0, 4.0]),
        )
        torch.testing.assert_close(transformed[:, 1], torch.zeros(2))

    def test_report_is_deterministic_and_records_zero_variance_indices(self):
        num = fit_column_standardizer(
            torch.tensor([[1.0, 25.0, 25.0], [3.0, 25.0, 25.0]])
        )
        style = fit_column_standardizer(
            torch.tensor([[0.1, 0.0], [0.2, 0.0]])
        )
        standardizers = {"num_prop": num, "llm_features": style}

        first = tabular_preprocessing_report(standardizers)
        second = tabular_preprocessing_report(standardizers)

        self.assertEqual(first, second)
        self.assertEqual(first["version"], TABULAR_PREPROCESSING_VERSION)
        self.assertEqual(first["num_prop"]["zero_variance_indices"], [1, 2])
        self.assertEqual(first["llm_features"]["zero_variance_indices"], [1])
        self.assertEqual(len(first["signature_sha256"]), 64)

    def test_nonfinite_values_are_sanitized_before_fit_and_transform(self):
        train = torch.tensor([[1.0, float("nan")], [3.0, float("inf")]])
        standardizer = fit_column_standardizer(train)
        transformed = apply_column_standardizer(train, standardizer)

        self.assertTrue(torch.isfinite(transformed).all())
        torch.testing.assert_close(transformed[:, 1], torch.zeros(2))

    def test_redacted_url_tokens_contribute_to_the_url_feature(self):
        row = {
            "posts": [
                {"text": "first <URL>"},
                {"text": "second <URL> and https://example.test/path"},
            ],
            "actions": [],
        }
        style = llm_style_features(row)

        self.assertAlmostEqual(float(style[5]), 1.5)


if __name__ == "__main__":
    unittest.main()
