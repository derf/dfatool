# NFP Model Generation for Peripherals and Software Product Lines

**dfatool** is a set of utilities for automated measurement of non-functional
properties of software product lines and embedded peripherals, and automatic
generation of NFP models based upon those.

Measurements and models for peripherals generally focus on energy and timing
behaviour expressed as a Priced Timed Automaton (PTA).

Measurements and models for software product lines focus on ROM/RAM usage and
may also include attributes such as throughput, latency, or energy.

## Energy Model Generation

to be documented.

## NFP Model Generation

### Running Benchmarks

**bin/explore-kconfig.py** works with any product line that supports kconfig and is capable of describing the non-functional properties of individual products.
To do so, it needs to support the **make**, **make clean**, **make randconfig**, **make nfpvalues** and **make nfpkeys** commands.
**make nfpvalues** is expected to print a JSON dict describing the non-functional property values of the current build;
**make nfpkeys** is expected to print a JSON dict with meta-data about those.
All of these commands can be changed, see `bin/explore-kconfig.py --help`.

See **examples/kconfig-static** for a simple example project, and [multipass](https://github.com/derf/multipass) and [kratos](https://ess.cs.uos.de/git/software/kratos/kratos) for more complex ones.
The `make_benchmark` section of **.gitlab-ci.yml** shows how to run benchmarks and generate a model for the example project.

As benchmark generation employs frequent recompilation, using a tmpfs is recommended.
Check out the product line (i.e., the benchmark target) into a directory on the tmpfs.
Next, create a second directory used for intermediate benchmark output, and `cd` to it.

Now, you can use `.../dfatool/bin/explore-kconfig.py` to benchmark the non-functional properties of various configurations, like so:

```
.../dfatool/bin/explore-kconfig.py --log-level debug --random 500 --with-neighbourhood .../my-project
```

This will benchmark 500 random configurations and additionaly explore the neighbourhood of each configuration by toggling boolean variables and exploring the range of int/hex variables.
Ternary features (y/m/n, as employed by the Linux kernel) are not supported.
The benchmark results (configurations and corresponding non-functional properties) are placed in the current working directory.

Once the benchmark is done, the observations can be compressed into a single file by running `.../dfatool/bin/analyze-kconfig.py --export-observations .../my-observations.json.xz --export-observations-only`.
Depending on the value of the **DFATOOL_KCONF_WITH_CHOICE_NODES** environment variable (see below), `choice` nodes are either treated as enum variables or groups of boolean variables.

### Generating Models

to be documented.

## Dependencies

Python 3.7 or newer with the following modules:

* frozendict
* matplotlib
* numpy
* scipy
* scikit-learn
* yaml
* zbar

On Debian Bullseye, all required modules are available as Debian packages.

## Code Style

Please only commit blackened code. It's best to check this with a pre-commit
hook:

```
#!/bin/sh

if git rev-parse --verify HEAD >/dev/null 2>&1
then
	against=HEAD
else
	# Initial commit: diff against an empty tree object
	against=4b825dc642cb6eb9a060e54bf8d69288fbee4904
fi

# Redirect output to stderr.
exec 1>&2

black --check $(git diff --cached --name-only --diff-filter=ACM $against | grep '\.py$')
```

## Environment Variables

The following variables may be set to alter the behaviour of dfatool components.

| Flag  | Range | Description |
| :--- | :---: | :--- |
| `DFATOOL_KCONF_WITH_CHOICE_NODES` | 0, **1** | Treat kconfig choices (e.g. "choice Model → MobileNet / ResNet / Inception") as enum parameters. If enabled, the corresponding boolean kconfig variables (e.g. "Model\_MobileNet") are not converted to parameters. If disabled, all (and only) boolean kconfig variables are treated as parameters. Mostly relevant for analyze-kconfig, eval-kconfig |
| `DFATOOL_COMPENSATE_DRIFT` | **0**, 1 | Perform drift compensation for loaders without sync input (e.g. EnergyTrace or Keysight) |
| `DFATOOL_DRIFT_COMPENSATION_PENALTY` | 0 .. 100 (default: majority vote over several penalties) | Specify penalty for ruptures.py PELT changepoint petection |
| `DFATOOL_DTREE_ENABLED` | 0, **1** | Use decision trees in get\_fitted |
| `DFATOOL_DTREE_FUNCTION_LEAVES` | 0, **1** | Use functions (fitted via linear regression) in decision tree leaves when modeling numeric parameters with at least three distinct values. If 0, integer parameters are treated as enums instead. |
| `DFATOOL_DTREE_SKLEARN_CART` | **0**, 1 | Use sklearn CART ("Decision Tree Regression") algorithm for decision tree generation. Uses binary nodes and supports splits on scalar variables. Overrides `FUNCTION_LEAVES` (=0) and `NONBINARY_NODES` (=0). |
| `DFATOOL_DTREE_SKLEARN_DECART` | **0**, 1 | Use sklearn CART ("Decision Tree Regression") algorithm for decision tree generation. Ignore scalar parameters, thus emulating the DECART algorithm. |
| `DFATOOL_DTREE_LMT` | **0**, 1 | Use [Linear Model Tree](https://github.com/cerlymarco/linear-tree) algorithm for regression tree generation. Uses binary nodes and linear functions. Overrides `FUNCTION_LEAVES` (=0) and `NONBINARY_NODES` (=0). |
| `DFATOOL_CART_MAX_DEPTH` | **0** .. *n* | maximum depth for sklearn CART. Default: unlimited. |
| `DFATOOL_USE_XGBOOST` | **0**, 1 | Use Extreme Gradient Boosting algorithm for decision forest generation. |
| `DFATOOL_XGB_N_ESTIMATORS` | 1 .. **100** .. *n* | Number of estimators (i.e., trees) for XGBoost. |
| `DFATOOL_XGB_MAX_DEPTH` | 2 .. **10** ** *n* | Maximum XGBoost tree depth. |
| `DFATOOL_KCONF_WITH_CHOICE_NODES` | 0, **1** | Generate enum parameters from kconfig choice nodes; ignore corresponding boolean config options. |
| `DFATOOL_KCONF_IGNORE_NUMERIC` | **0**, 1 | Ignore numeric (int/hex) configuration options. Useful for comparison with CART/DECART. |
| `DFATOOL_KCONF_IGNORE_STRING` | **0**, 1 | Ignore string configuration options. Useful for comparison with CART/DECART. |
| `DFATOOL_FIT_LINEAR_ONLY` | **0**, 1 | Only consider linear functions (a + bx) in regression analysis. Useful for comparison with Linear Model Trees / M5. |
| `DFATOOL_REGRESSION_SAFE_FUNCTIONS` | **0**, 1 | Use safe functions only (e.g. 1/x returnning 1 for x==0) |
| `DFATOOL_DTREE_NONBINARY_NODES` | 0, **1** | Enable non-binary nodes (i.e., nodes with more than two children corresponding to enum variables) in decision trees |
| `DFATOOL_DTREE_IGNORE_IRRELEVANT_PARAMS` | 0, **1** | Ignore parameters deemed irrelevant by stddev heuristic during regression tree generation |
| `DFATOOL_DTREE_LOSS_IGNORE_SCALAR` | **0**, 1 | Ignore scalar parameters when computing the loss for split node candidates. Instead of computing the loss of a single partition for each `x_i == j`, compute the loss of partitions for `x_i == j` in which non-scalar parameters vary and scalar parameters are constant. This way, scalar parameters do not affect the decision about which non-scalar parameter to use for splitting. |
| `DFATOOL_PARAM_CATEGORIAL_TO_SCALAR` | **0**, 1 | Some models (e.g. FOL, sklearn CART, XGBoost) do not support categorial parameters. Ignore them (0) or convert them to scalar indexes (1). |
| `DFATOOL_FIT_FOL` | **0**, 1 | Build a first-order linear function (i.e., a * param1 + b * param2 + ...) instead of more complex functions or tree structures. |
| `DFATOOL_FOL_SECOND_ORDER` || **0**, 1 | Add second-order components (interaction of feature pairs) to first-order linear function. |

## Examples

Suitable for [kconfig-webconf](https://ess.cs.uos.de/git-build/kconfig-webconf/master/)

### Toy Example

The NFP values should be exactly as described by the selected configuration options.

* [Kconfig](https://ess.cs.uos.de/git-build/dfatool/main/example-static.kconfig)
* [CART](https://ess.cs.uos.de/git-build/dfatool/main/example-static-cart-b.json) without choice nodes
* [CART](https://ess.cs.uos.de/git-build/dfatool/main/example-static-cart-nb.json) with choice nodes
* [RMT](https://ess.cs.uos.de/git-build/dfatool/main/example-static-rmt-b.json) without choice nodes
* [RMT](https://ess.cs.uos.de/git-build/dfatool/main/example-static-rmt-nb.json) with choice nodes

### x264 Video Encoding

* [Kconfig](https://ess.cs.uos.de/git-build/dfatool/master/x264.kconfig)
* [CART](https://ess.cs.uos.de/git-build/dfatool/master/x264-cart.json)
* [RMT](https://ess.cs.uos.de/git-build/dfatool/master/x264-rmt.json)
