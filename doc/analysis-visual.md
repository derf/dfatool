# Visual Data Analysis

The parameter and NFP filters from [Textual Data Analysis](analysis-textual.md) apply here as well.

## Raw Data Visualization

There are two ways of visualizing all measured data independent of their parameters:

* `--boxplot-unparam PREFIX` writes boxplots of all observations to PREFIX(name)-(attribute).pdf and combined boxplots to PREFIX(name).pdf. These may be helpful to see which observations are stable and which show a lot of variance, possibly due to the influence of parameters.
* `--plot-unparam=name:attribute:ylabel` plots all observations of name/attribute in the order in which they were observed. Useful to identify trends (especially when the parameter variation scheme is known as well) and interference.

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

[!](/media/n_dpus-dpu_alloc-1.png)
