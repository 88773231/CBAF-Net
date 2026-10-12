import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_confusion import (
    configure_style,
    draw_panel,
    fit_cell_annotations,
)


class ConfusionFigureLayoutTests(unittest.TestCase):
    def test_percentage_labels_fit_without_changing_values(self):
        configure_style()
        figure = plt.figure(figsize=(7.12, 5.72))
        grid = figure.add_gridspec(
            2, 3, width_ratios=(1.0, 1.0, 0.045),
            left=0.115, right=0.895, bottom=0.105, top=0.855,
            wspace=0.42, hspace=0.50,
        )
        axes = []
        for index, task in enumerate(("Twibot22", "Quadbot")):
            classes = 3 if task == "Twibot22" else 4
            matrix = np.eye(classes)
            axis = figure.add_subplot(grid[index, 0])
            draw_panel(axis, matrix, task, "CBAF-Net", str(index))
            axes.append((axis, matrix))
        figure.canvas.draw()
        for axis, matrix in axes:
            texts_before = [label.get_text() for label in axis.texts]
            fit_cell_annotations(axis)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            self.assertEqual(texts_before, [label.get_text() for label in axis.texts])
            np.testing.assert_array_equal(axis.images[0].get_array(), matrix)
            for label in axis.texts:
                column, row = label.get_position()
                first = axis.transData.transform((column - 0.45, row - 0.45))
                last = axis.transData.transform((column + 0.45, row + 0.45))
                bounds = label.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0 + 0.5, min(first[0], last[0]))
                self.assertLessEqual(bounds.x1 - 0.5, max(first[0], last[0]))
                self.assertGreaterEqual(bounds.y0 + 0.5, min(first[1], last[1]))
                self.assertLessEqual(bounds.y1 - 0.5, max(first[1], last[1]))
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
