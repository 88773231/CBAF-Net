import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from dataset_pipeline.export_relative_timing import gap_statistics, relative_timing


class RelativeTimingTests(unittest.TestCase):
    def row(self, offset=0, gaps=None):
        origin = datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset)
        elapsed = 0
        posts = []
        for index in range(25):
            if index:
                elapsed += gaps[index - 1] if gaps else 60
            posts.append({
                "time_step": index // 5 + 1,
                "created_at": (origin + timedelta(seconds=elapsed)).isoformat(),
            })
        return {"posts": posts}

    def test_absolute_date_shift_does_not_change_features(self):
        left, left_steps = relative_timing(self.row())
        right, right_steps = relative_timing(self.row(offset=86400 * 365))
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left_steps, right_steps)

    def test_label_provenance_and_content_are_not_inputs(self):
        row = self.row()
        expected = relative_timing(row)
        row.update({"label": 3, "provenance": {"generator": "private"}, "profile": {}})
        for post in row["posts"]:
            post["text"] = "arbitrary content"
        actual = relative_timing(row)
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    def test_regular_and_irregular_intervals_are_distinguishable(self):
        regular, _ = relative_timing(self.row())
        irregular, _ = relative_timing(self.row(gaps=[5, 115] * 12))
        self.assertEqual(float(regular[3]), 0)
        self.assertGreater(float(irregular[3]), 0)

    def test_missing_or_timezone_free_timestamps_fail_closed(self):
        row = self.row()
        row["posts"][0]["created_at"] = ""
        with self.assertRaises(ValueError):
            relative_timing(row)
        row["posts"][0]["created_at"] = "2022-01-01T00:00:00"
        with self.assertRaises(ValueError):
            relative_timing(row)

    def test_subsecond_timestamp_precision_is_not_a_feature(self):
        np.testing.assert_array_equal(
            gap_statistics([0, 60, 120, 180]), gap_statistics([0.001, 60.002, 120.003, 180.004])
        )


if __name__ == "__main__":
    unittest.main()
