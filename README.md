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
* LightGBM: Regressin Forests
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
| `DFATOOL_MODEL` | cart, decart, fol, lgbm, lmt, **rmt**, symreg, uls, xgb | Modeling method. See below for method-specific configuration options. |
| `DFATOOL_RMT_SUBMODEL` | cart, fol, static, symreg, **uls** | Modeling method for RMT leaf functions. |
| `DFATOOL_CART_MAX_DEPTH` | **0** .. *n* | maximum depth for sklearn CART. Default (0): unlimited. |
| `DFATOOL_LGBM_BOOSTER` | **gbdt**, dart, rf | Boosting type. |
| `DFATOOL_LGBM_N_ESTIMATORS` | .., **100**, .. | Number of estimators. |
| `DFATOOL_LGBM_MAX_DEPTH` | **-1**, .., *n* | Maximum tree depth, unlimited if ≤ 0. |
| `DFATOOL_LGBM_NUM_LEAVES` | .., **31**, .. | Maximum number of leaves per tree. |
| `DFATOOL_LGBM_SUBSAMPLE` | 0.0 .. **1.0** | Subsampling ration. |
| `DFATOOL_LGBM_LEARNING_RATE` | 0 .. **0.1** .. 1 | Learning rate. |
| `DFATOOL_LGBM_MIN_SPLIT_GAIN` | **0.0** .. 1 | Minimum loss reduction required for a split. |
| `DFATOOL_LGBM_MIN_CHILD_SAMPLES` | .., **20**, .. | Minimum samples that each leaf of a split candidate must contain. |
| `DFATOOL_LGBM_REG_ALPHA` | **0.0** .. *n* | L1 regularization term on weights. |
| `DFATOOL_LGBM_REG_LAMBDA` | **0.0** .. *n* | L2 regularization term on weights. |
| `DFATOOL_LMT_MAX_DEPTH` | **5** .. 20 | Maximum depth for LMT. |
| `DFATOOL_LMT_MIN_SAMPLES_SPLIT` | 0.0 .. 1.0, **6** .. *n* | Minimum samples required to still perform an LMT split. A value below 1.0 sets the specified ratio of the total number of training samples as minimum. |
| `DFATOOL_LMT_MIN_SAMPLES_LEAF` | 0.0 .. **0.1** .. 1.0, 3 .. *n* | Minimum samples that each leaf of a split candidate must contain. A value below 1.0 specifies a ratio of the total number of training samples. A value above 1 specifies an absolute number of samples. |
| `DFATOOL_LMT_MAX_BINS` | 10 .. **120** | Number of bins used to determine optimal split. LMT default: 25. |
| `DFATOOL_LMT_CRITERION` | **mse**, rmse, mae, poisson | Error metric to use when selecting best split. |
| `DFATOOL_ULS_ERROR_METRIC` | **ssr**, rmsd, mae, … | Error metric to use when selecting best-fitting function during unsupervised least squares (ULS) regression. Least squares regression itself minimzes root mean square deviation (rmsd), hence the equivalent (but partitioning-compatible) sum of squared residuals (ssr) is the default. Supports all metrics accepted by `--error-metric`. |
| `DFATOOL_ULS_MIN_DISTINCT_VALUES` | 2 .. **3** .. *n* | Minimum number of unique values a parameter must take to be eligible for ULS |
| `DFATOOL_ULS_SKIP_CODEPENDENT_CHECK` | **0**, 1 | Do not detect and remove co-dependent features in ULS. |
| `DFATOOL_XGB_N_ESTIMATORS` | 1 .. **100** .. *n* | Number of estimators (i.e., trees) for XGBoost. |
| `DFATOOL_XGB_MAX_DEPTH` | 2 .. **6** .. *n* | Maximum XGBoost tree depth. |
| `DFATOOL_XGB_SUBSAMPLE` | 0.0 .. **1.0** | XGBoost subsampling ratio. |
| `DFATOOL_XGB_ETA` | 0 .. **0.3** .. 1 | XGBoost learning rate (shrinkage). |
| `DFATOOL_XGB_GAMMA` | **0.0** .. *n* | XGBoost minimum loss reduction required to to make a further partition on a leaf node. |
| `DFATOOL_XGB_REG_ALPHA` | **0.0** .. *n* | XGBoost L1 regularization term on weights. |
| `DFATOOL_XGB_REG_LAMBDA` | 0 .. **1** .. *n* | XGBoost L2 regularization term on weights. |
| `OMP_NUM_THREADS` | *number of CPU cores* | Maximum number of threads used per XGBoost learner. A limit of 4 threads appears to be ideal. Note that dfatool may spawn several XGBoost instances at the same time. |
| `DFATOOL_KCONF_IGNORE_NUMERIC` | **0**, 1 | Ignore numeric (int/hex) configuration options. Useful for comparison with CART/DECART. |
| `DFATOOL_KCONF_IGNORE_STRING` | 0, **1** | Ignore string configuration options. These often hold compiler paths and other not really helpful information. |
| `DFATOOL_REGRESSION_SAFE_FUNCTIONS` | **0**, 1 | Use safe functions only (e.g. 1/x returnning 1 for x==0) |
| `DFATOOL_RMT_NONBINARY_NODES` | 0, **1** | Enable non-binary nodes (i.e., nodes with more than two children corresponding to enum variables) in decision trees |
| `DFATOOL_RMT_RELEVANCE_METHOD` | **none**, std\_by\_param | Ignore parameters deemed irrelevant by the specified heuristic during regression tree generation. Use with caution. |
| `DFATOOL_PARAM_RELEVANCE_THRESHOLD` | 0 .. **0.5** .. 1 | Threshold for relevant parameter detection: parameter *i* is relevant if mean standard deviation (data partitioned by all parameters) / mean standard deviation (data partition by all parameters but *i*) is less than threshold |
| `DFATOOL_RMT_LOSS_IGNORE_SCALAR` | **0**, 1 | Ignore scalar parameters when computing the loss for split node candidates. Instead of computing the loss of a single partition for each `x_i == j`, compute the loss of partitions for `x_i == j` in which non-scalar parameters vary and scalar parameters are constant. This way, scalar parameters do not affect the decision about which non-scalar parameter to use for splitting. |
| `DFATOOL_PARAM_CATEGORICAL_TO_SCALAR` | **0**, 1 | Some models (e.g. FOL, sklearn CART, XGBoost) do not support categorical parameters. Ignore them (0) or convert them to scalar indexes (1). Conversion uses lexical order. |
| `DFATOOL_FOL_SECOND_ORDER` | **0**, 1 | Add second-order components (interaction of feature pairs) to first-order linear function. |

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
