# Visual Data Analysis

The parameter and NFP filters from [Textual Data Analysis](analysis-textual.md) apply here as well.

## Raw Data Visualization

There are two ways of visualizing all measured data independent of their parameters:

* `--boxplot PREFIX` writes boxplots of all observations to PREFIX/(name)-(attribute).pdf. These may be helpful to see which observations are stable and which show a lot of variance, possibly due to the influence of parameters.
* `--plot-unparam=name:attribute:ylabel` plots all observations of name/attribute in the order in which they were observed. Useful to identify trends, especially when the parameter variation scheme is known as well.
