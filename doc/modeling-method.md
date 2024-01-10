# Modeling Method Selection

## CART (Regression Trees)

Enable these with `DFATOOL_DTREE_SKLEARN_CART=1` and `--force-tree`.

### Related Options

* `DFATOOL_PARAM_CATEGORIAL_TO_SCALAR=1` converts categorial parameters (which are not supported by CART) to numeric ones.

## XGB (Gradient-Boosted Forests / eXtreme Gradient boosting)

Enable these with `DFATOOL_USE_XGBOOST=1` and `--force-tree`.
You should also specify `DFATOOL_XGB_N_ESTIMATORS`, `DFATOOL_XGB_MAX_DEPTH`, and possibly `OMP_NUM_THREADS`.

### Related Options

* `DFATOOL_PARAM_CATEGORIAL_TO_SCALAR=1` converts categorial parameters (which are not supported by XGB) to numeric ones.
* Anything prefixed with `DFATOOL_XGB_`.

## LMT (Linear Model Trees)

Enable these with `DFATOOL_DTREE_LMT=1` and `--force-tree`.
They always use a maximum depth of 20.

### Related Options

* `DFATOOL_PARAM_CATEGORIAL_TO_SCALAR=1` converts categorial parameters (which are not supported by LMT) to numeric ones.

## RMT (Regression Model Trees)

This is the default modeling method.
It first uses a heuristic to determine which parameters are relevant, and then builds a non-binary decision tree with function leaves to predict how those affect the performance attribute.
Depending on the set of relevant parameters and how well they can be modeled with tree and function structures, it may output a simple function (without tree structure), a tree with static leaves, or a single static value that does not take any parameters into account.
All of these are valid regression model trees.

### Related Options

* `--force-tree` builds a tree structure even if dfatool's heuristic indicates that no non-integer parameter affects the modeled performance attribute.
* `DFATOOL_DTREE_IGNORE_IRRELEVANT_PARAMS=0` disables the relevant parameter detection heuristic when building the tree structure. By default, irrelevant parameters cannot end up as decision nodes.
* `DFATOOL_FIT_LINEAR_ONLY=1` makes RMT behave more like LMT by only considering linear functions in leaf nodes.
* `DFATOOL_SKIP_CODEPENDENT_CHECK=1`
* `DFATOOL_REGRESSION_SAFE_FUNCTIONS=1`

##


* CART: Regression Trees
* DECART: Regression Trees with exclusively binary features/parameters
* XGB: Regression Forests
* LMT: Linear Model Trees
* RMT: [Regression Model Trees](https://ess.cs.uos.de/static/papers/Friesel-2022-CPSIoTBench.pdf) with [non-binary nodes](https://ess.cs.uos.de/static/papers/Friesel-2022-CAIN.pdf)
* Least-Squares Regression
