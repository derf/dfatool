# Performance Model Generation for Peripherals and Software Product Lines

**dfatool** is a set of utilities for automated performance model generation.
It supports a wide range of application domains, including

* unattened energy measurements and energy model generation for embedded peripherals,
* unattended performance (“non-functional property”) measurements and NFP model generation for software product lines and SPL-like software projects, and
* data analysis and performance model generation for arbitrary data sets out of text-based log files.

Measurements and models for peripherals (`generate-dfa-benchmark.py` and `analyze-archive.py`) generally focus on energy and timing behaviour expressed as a [Priced Timed Automaton (PTA)](https://ess.cs.uos.de/static/papers/Friesel_2018_sies.pdf) with [Regression Model Trees (RMT)](https://ess.cs.uos.de/static/papers/Friesel-2022-CPSIoTBench.pdf).

Measurements and models for software product lines (`explore-kconfig.py` and `analyze-kconfig.py`) focus on ROM/RAM usage and may also include attributes such as throughput, latency, or energy.
The variability model of the software product line must be expressed in the [Kconfig language](https://www.kernel.org/doc/Documentation/kbuild/kconfig-language.txt).
Generated models can be used with [kconfig-webconf](https://ess.cs.uos.de/git/software/kconfig-webconf).
This allows for [Retrofitting Performance Models onto Kconfig-based Software Product Lines](https://ess.cs.uos.de/static/papers/Friesel-2022-SPLC.pdf).

Models for arbitrary other kinds of configurable components (`analyze-log.py`) rely on logfiles that contain "`[::]` *Key* *Attribute* | *parameters* | *NFP values*" lines.
Here, only analysis and model generation are automated, and users have to generate the logfiles by themselves.

The name **dfatool** comes from the fact that benchmark generation for embedded peripherals relies on a deterministic finite automaton (DFA) that specifies the peripheral's behaviour (i.e., states and transitions caused by driver functions or signalled by interrupts).
It is meaningless in the context of software product lines and other configurable components.

The remainder of this README references domain-specific data acquisition and model generation how-tos as well as dependencies and global options and environment variables.

## Data Acquisition

* [Measuring non-functional properties ("performance attributes") of software product lines](doc/acquisition-nfp.md)

Legacy documentation; may be outdated:
* [Energy Benchmarks with Kratos](doc/energy-kratos.md) (DE)
* [Energy Benchmarks with Multipass](doc/energy-multipass.md) (DE)
* [Performance Benchmarks for Multipass](doc/nfp-multipass.md) (DE)

## Data Analysis

It can be helpful to visualize acquired data points to get a feel for how the observed performance attributes behave.
Most of the options and methods documented here work for all three scripts: analyze-archive, analyze-kconfig, and analyze-log.

* [Textual Data Analysis](doc/analysis-textual.md)
* [Visual Data Analysis](doc/analysis-visual.md)

## Model Generation

dfatool supports six types of performance models:

* CART: Regression Trees
* DECART: Regression Trees with exclusively binary features/parameters
* XGB: Regression Forests
* LMT: Linear Model Trees
* RMT: [Regression Model Trees](https://ess.cs.uos.de/static/papers/Friesel-2022-CPSIoTBench.pdf) with [non-binary nodes](https://ess.cs.uos.de/static/papers/Friesel-2022-CAIN.pdf)
* Least-Squares Regression

Least-Squares Regression is essentially a subset of RMT with just a single tree node.
LMT and RMT differ significantly, as LMT uses a learning algorithm that starts out with a DECART and uses bottom-up pruning to turn it into an LMT, whereas RMT build a DECART that only considers parameters that are not suitable for least-squares regression and then uses least-squares regression to find and fit leaf functions.

By default, dfatool uses heuristics to determine whether it should generate a simple least-squares regression function or a fully-fledged RMT.
Arguments such as `--force-tree` and environment variables (below) can be used to generate a different flavour of performance model; see [Modeling Method Selection](doc/modeling-method.md).
Again, most of the options and methods documented here work for all three scripts: analyze-archive, analyze-kconfig, and analyze-log.

* [Model Visualization and Export](doc/model-visual.md)
* [Modeling Method Selection](doc/modeling-method.md)
* [Assessing Model Quality](doc/model-assessment.md)

## Model Application

Legacy documentation; may be outdated:
* [Generating performance models for software product lines](doc/analysis-nfp.md)
* [Data Analysis and Performance Model Generation from Log Files](doc/analysis-logs.md)

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
| `DFATOOL_CART_MAX_DEPTH` | **0** .. *n* | maximum depth for sklearn CART. Default (0): unlimited. |
| `DFATOOL_ULS_ERROR_METRIC` | **rmsd**, mae, p50, p90 | Error metric to use when selecting best-fitting function during unsupervised least squares (ULS) regression. Least squares regression itself minimzes root mean square deviation (rmsd), hence rmsd is the default. |
| `DFATOOL_ULS_MIN_DISTINCT_VALUES` | 2 .. **3** .. *n* | Minimum number of unique values a parameter must take to be eligible for ULS |
| `DFATOOL_SKIP_CODEPENDENT_CHECK` | **0**, 1 | Do not detect and remove co-dependent features in ULS. |
| `DFATOOL_USE_XGBOOST` | **0**, 1 | Use Extreme Gradient Boosting algorithm for decision forest generation. |
| `DFATOOL_XGB_N_ESTIMATORS` | 1 .. **100** .. *n* | Number of estimators (i.e., trees) for XGBoost. Mandatory. |
| `DFATOOL_XGB_MAX_DEPTH` | 2 .. **10** .. *n* | Maximum XGBoost tree depth. XGBoost default: 6 |
| `DFATOOL_XGB_SUBSAMPLE` | 0 .. **0.7** .. 1 | XGBoost subsampling ratio. XGBoost default: 1 |
| `DFATOOL_XGB_ETA` | 0 .. **0.3** .. 1 | XGBoost learning rate (shrinkage). XGboost default: 0.3 |
| `DFATOOL_XGB_GAMMA` | 0 .. **0.01** .. *n* | XGBoost minimum loss reduction required to to make a further partition on a leaf node. XGBoost default: 0 |
| `DFATOOL_XGB_REG_ALPHA` | 0 .. **0.0006** .. *n* | XGBoost L1 regularization term on weights. |
| `DFATOOL_XGB_REG_LAMBDA` | 0 .. **1** .. *n* | XGBoost L2 regularization term on weight. |
| `OMP_NUM_THREADS` | *number of CPU cores* | Maximum number of threads used per XGBoost learner. A limit of 4 threads appears to be ideal. Note that dfatool may spawn several XGBoost instances at the same time. |
| `DFATOOL_KCONF_IGNORE_NUMERIC` | **0**, 1 | Ignore numeric (int/hex) configuration options. Useful for comparison with CART/DECART. |
| `DFATOOL_KCONF_IGNORE_STRING` | 0, **1** | Ignore string configuration options. These often hold compiler paths and other not really helpful information. |
| `DFATOOL_FIT_LINEAR_ONLY` | **0**, 1 | Only consider linear functions (a + bx) in regression analysis. Useful for comparison with Linear Model Trees / M5. |
| `DFATOOL_REGRESSION_SAFE_FUNCTIONS` | **0**, 1 | Use safe functions only (e.g. 1/x returnning 1 for x==0) |
| `DFATOOL_DTREE_NONBINARY_NODES` | 0, **1** | Enable non-binary nodes (i.e., nodes with more than two children corresponding to enum variables) in decision trees |
| `DFATOOL_DTREE_IGNORE_IRRELEVANT_PARAMS` | **0**, 1 | Ignore parameters deemed irrelevant by stddev heuristic during regression tree generation. Use with caution. |
| `DFATOOL_PARAM_RELEVANCE_THRESHOLD` | 0 .. **0.5** .. 1 | Threshold for relevant parameter detection: parameter *i* is relevant if mean standard deviation (data partitioned by all parameters) / mean standard deviation (data partition by all parameters but *i*) is less than threshold |
| `DFATOOL_DTREE_LOSS_IGNORE_SCALAR` | **0**, 1 | Ignore scalar parameters when computing the loss for split node candidates. Instead of computing the loss of a single partition for each `x_i == j`, compute the loss of partitions for `x_i == j` in which non-scalar parameters vary and scalar parameters are constant. This way, scalar parameters do not affect the decision about which non-scalar parameter to use for splitting. |
| `DFATOOL_PARAM_CATEGORIAL_TO_SCALAR` | **0**, 1 | Some models (e.g. FOL, sklearn CART, XGBoost) do not support categorial parameters. Ignore them (0) or convert them to scalar indexes (1). |
| `DFATOOL_FIT_FOL` | **0**, 1 | Build a first-order linear function (i.e., a * param1 + b * param2 + ...) instead of more complex functions or tree structures. Must not be combined with `--force-tree`. |
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
