# Method figure

`cbaf_architecture.svg` is the editable source for manuscript Figure 1.
`cbaf_architecture.png` is its 3760 by 2480 pixel rendering on a white
background. Use a full-width placement, not a single-column reduction.

The drawing is a schematic of the released implementation, not an additional
experiment. Profile, post, event/action, numerical, and style inputs enter
the fixed BotDMM backbone; numerical and style fields have separate linear
encoders. Numerical and style fields also enter the
independently fitted behavioral-statistics expert. The same-profile exclusion
applies to the split-local kNN graph; this is not an observed follower or
reply network. Only output probabilities are combined, using the global
validation-selected weight locked before test evaluation.

People, the phone display, messages, calendar, robots, and tool symbols are
original vector drawings. The content MLP, directed neighbor context, temporal
features, and HGB trees illustrate implemented computations. Layer widths,
drawn neighbor/tree counts, and the probability bars do not specify actual
tensor dimensions, ten-neighbor counts, estimator size, or experimental
measurements. The phone's message snippets and the validation curve are also
illustrative.

Visual composition was informed by the author-provided figures from *The
intelligent social event observer: Multi-source continuous event integration,
discovery, and induction with LLMs* and Figure 2 of *CACL: Community-Aware
Heterogeneous Graph Contrastive Learning for Social Media Bot Detection*
(Findings of ACL 2024). The new drawing uses general scene-based presentation,
not copied logos, artwork, layouts, or model modules from those papers.
Watermarked stock icon references were not incorporated.

The figure update changes neither the detector nor the released predictions.
The source assets are covered by the repository's code/documentation license.

## Ablation figure

`cbaf_ablation.png` and `cbaf_ablation.svg` are the current manuscript
renderings of `analysis/plot_ablation.py`. They read the audited means from
`results/strict_postprocessed/strict_rerun_summary.json`; only title, subtitle,
legend, and panel spacing have changed from the archived rendering.
`tests/test_ablation_figure_layout.py` checks text clearance and the reported
value annotations. The underlying results and archived release remain intact.

## Confusion figure

`cbaf_confusion.png` and `cbaf_confusion.svg` are the current manuscript
renderings of `analysis/plot_confusion.py`. They use the unchanged public
test predictions from `results/strict_postprocessed/predictions/`.
Percentage annotations are fitted to their rendered cells at print width;
the matrices, normalization, class labels, and values are unchanged.
`tests/test_confusion_figure_layout.py` verifies the annotation bounds and
numeric invariance. The original `v1.1.0` confusion assets remain unchanged
inside the immutable result archive.
