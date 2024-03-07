# Modeling Method Selection

Set `DFATOOL_MODEL` to an appropriate value, e.g. `DFATOOL_MODEL=cart`.

## CART (Regression Trees)

sklearn CART ("Decision Tree Regression") algorithm. Uses binary nodes and supports splits on scalar variables.

### Related Options

* `DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1` converts categorical parameters (which are not supported by CART) to numeric ones.

## DECART (Regression Trees)

sklearn CART ("Decision Tree Regression") algorithm. Ignores scalar parameters, thus emulating the DECART algorithm.

## FOL (First-Order Linear function)

Build a first-order linear function (i.e., a * param1 + b * param2 + ...).

## LMT (Linear Model Trees)

[Linear Model Tree](https://github.com/cerlymarco/linear-tree) algorithm. Uses binary nodes and linear functions.

### Related Options

See the [LinearTreeRegressor documentation](lib/lineartree/lineartree.py) for details on training hyper-parameters.

* `DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1` converts categorical parameters (which are not supported by LMT) to numeric ones.
* `DFATOOL_LMT_MAX_DEPTH`
* `DFATOOL_LMT_MIN_SAMPLES_SPLIT`
* `DFATOOL_LMT_MIN_SAMPLES_LEAF`
* `DFATOOL_LMT_MAX_BINS`
* `DFATOOL_LMT_CRITERION`

## RMT (Regression Model Trees)

This is the default modeling method.
It first uses a heuristic to determine which parameters are relevant, and then builds a non-binary decision tree with function leaves to predict how those affect the performance attribute.
Depending on the set of relevant parameters and how well they can be modeled with tree and function structures, it may output a simple function (without tree structure), a tree with static leaves, or a single static value that does not take any parameters into account.
All of these are valid regression model trees.

### Related Options

* `--force-tree` builds a tree structure even if dfatool's heuristic indicates that no non-integer parameter affects the modeled performance attribute.
* `DFATOOL_RMT_IGNORE_IRRELEVANT_PARAMS=0` disables the relevant parameter detection heuristic when building the tree structure. By default, irrelevant parameters cannot end up as decision nodes.
* `DFATOOL_RMT_SUBMODEL=fol` makes RMT only consider linear functions (a + bx) in regression analysis. Useful for comparison with LMT / M5.
* `DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1`
* `DFATOOL_ULS_SKIP_CODEPENDENT_CHECK=1`
* `DFATOOL_REGRESSION_SAFE_FUNCTIONS=1`

## XGB (Gradient-Boosted Forests / eXtreme Gradient boosting)

You should also specify `DFATOOL_XGB_N_ESTIMATORS`, `DFATOOL_XGB_MAX_DEPTH`, and possibly `OMP_NUM_THREADS`.

### Related Options

* `DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1` converts categorical parameters (which are not supported by XGB) to numeric ones.
* Anything prefixed with `DFATOOL_XGB_`.

## Least-Squares Regression

If dfatool determines that there is no need for a tree structure, or if `DFATOOL_MODEL=uls`, it will go straight to least-squares regression.
By default, it still utilizes the RMT/ULS algorithms to find and fit a suitable function template.
If needed, `--function-override` can be used to set a function template manually.
For instance, in order to specify that NMC DPU allocation latency is a function of the number of DPUs (and nothing else), ue `--function-override 'NMC reconfiguration:latency_dpu_alloc_us:regression_arg(0) + regression_arg(1) * parameter(n_dpus)'`

* CART: Regression Trees
* DECART: Regression Trees with exclusively binary features/parameters
* XGB: Regression Forests
* LMT: Linear Model Trees
* RMT: [Regression Model Trees](https://ess.cs.uos.de/static/papers/Friesel-2022-CPSIoTBench.pdf) with [non-binary nodes](https://ess.cs.uos.de/static/papers/Friesel-2022-CAIN.pdf)
* Least-Squares Regression
