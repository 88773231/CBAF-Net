import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.plot_ablation import build_figure, load_results


ROOT = Path(__file__).resolve().parents[1]


class AblationFigureLayoutTests(unittest.TestCase):
    def setUp(self):
        self.results = load_results(
            ROOT / "results" / "strict_postprocessed" / "strict_rerun_summary.json"
        )
        self.figure = build_figure(self.results)
        self.figure.canvas.draw()
        self.renderer = self.figure.canvas.get_renderer()

    def tearDown(self):
        plt.close(self.figure)

    def test_header_and_panel_titles_have_clearance(self):
        subtitle = next(
            text for text in self.figure.texts
            if text.get_text().startswith("Mean test Macro-F1")
        )
        title_box = self.figure._suptitle.get_window_extent(self.renderer)
        subtitle_box = subtitle.get_window_extent(self.renderer)
        legend_box = self.figure.legends[0].get_window_extent(self.renderer)
        self.assertGreaterEqual(title_box.y0 - subtitle_box.y1, 2.0)
        self.assertGreaterEqual(subtitle_box.y0 - legend_box.y1, 2.0)
        for axes in self.figure.axes:
            panel_box = axes.title.get_window_extent(self.renderer)
            self.assertGreaterEqual(legend_box.y0 - panel_box.y1, 2.0)

    def test_annotations_preserve_audited_means(self):
        for axes, values in zip(self.figure.axes, self.results.values()):
            self.assertEqual(
                [text.get_text() for text in axes.texts],
                [f"{value:.4f}" for value in values],
            )
            lower, upper = axes.get_xlim()
            self.assertTrue(all(lower < value < upper for value in values))


if __name__ == "__main__":
    unittest.main()
