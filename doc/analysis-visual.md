# Visual Data Analysis

The parameter and NFP filters from [Textual Data Analysis](analysis-textual.md) apply here as well.

## Raw Data Visualization

There are two ways of visualizing all measured data independent of their parameters:

* `--boxplot-unparam PREFIX` writes boxplots of all observations to PREFIX(name)-(attribute).pdf and combined boxplots to PREFIX(name).pdf.
  These may be helpful to see which observations are stable and which show a lot of variance, possibly due to the influence of parameters.
  By default, the boxplots are also shown interactively; use `--non-interactive` to display that.
* `--plot-unparam=name:attribute:ylabel` plots all observations of name/attribute in the order in which they were observed.
  Useful to identify trends (especially when the parameter variation scheme is known as well) and interference.
  The plot is shown interactively, but not written to the filesystem.

## Influence of a single Non-Functional Property on a Performance Attribute

Assume that we want to see how the number of requested UPMEM DPUs (`n_dpus`)
influences the latency of its `dpu_alloc` call (`latency_dpu_alloc_us`). In
dfatool terms, this means that we want to visualize the influence of the
parameter (or NFP) `n_dpus` on the attribute `latency_dpu_alloc_us`.

`--plot-param 'NMC reconfiguration:latency_dpu_alloc_us:n_dpus'` does just
that. "NMC reconfiguration" is the benchmark key, followed by attribute and
parameter name. It shows each distinct configuration (parameter/NFP value) in
a different colour. Combining it with `--filter-param` and `--ignore-param`
may help de-clutter the plot.

dfatool will additionally plot the predicted performance for each distinct
configuration as a solid line. The plot is saved to (name)-(attribute)-(parameter).pdf
and shown interactively unless `--non-interactive` has been specified.

![](/media/n_dpus-dpu_alloc-1.png)
