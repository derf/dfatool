# Performance Model Generation for Peripherals and Software Product Lines

**dfatool** is a set of utilities for automated measurement of non-functional properties of software product lines and embedded peripherals, and automatic generation of performance models based upon those.

Measurements and models for peripherals (`generate-dfa-benchmark.py` and `analyze-archive.py`) generally focus on energy and timing behaviour expressed as a [Priced Timed Automaton (PTA)](https://ess.cs.uos.de/static/papers/Friesel_2018_sies.pdf) with [Regression Model Trees (RMT)](https://ess.cs.uos.de/static/papers/Friesel-2022-CPSIoTBench.pdf).

Measurements and models for software product lines (`explore-kconfig.py` and `analyze-kconfig.py`) focus on ROM/RAM usage and may also include attributes such as throughput, latency, or energy.
The variability model of the software product line must be expressed in the [Kconfig language](https://www.kernel.org/doc/Documentation/kbuild/kconfig-language.txt).
Generated models can be used with [kconfig-webconf](https://ess.cs.uos.de/git/software/kconfig-webconf).
This allows for [Retrofitting Performance Models onto Kconfig-based Software Product Lines](https://ess.cs.uos.de/static/papers/Friesel-2022-SPLC.pdf).

Models for arbitrary other kinds of configurable components (`analyze-log.py`) are also supported and work with logfiles that contain "`[::]` *Key* *Attribute* | *parameters* | *NFP values*" lines.
Here, only analysis and model generation are automated, and users have to generate the logfiles by themselves.

The name **dfatool** comes from the fact that benchmark generation for embedded peripherals relies on a deterministic finite automaton (DFA) that specifies the peripheral's behaviour (i.e., states and transitions caused by driver functions or signalled by interrupts).
It is meaningless in the context of software product lines and other configurable components.

## Data Acquisition

* [Measuring non-functional properties ("performance attributes") of software product lines](doc/acquisition-nfp)

Legacy documentation; may be outdated:
* [Energy Benchmarks with Kratos](doc/energy-kratos) (DE)
* [Energy Benchmarks with Multipass](doc/energy-multipass) (DE)
* [Performance Benchmarks for Multipass](doc/nfp-multipass) (DE)

## Model Generation

* [Generating performance models for software product lines](doc/analysis-nfp)

## Log-Based Performance Model Generation

Here, dfatool works with lines of the form "`[::]` *Key* *Attribute* | *parameters* | *NFP values*", where *parameters* is a space-separated series of *param=value* entries (i.e., benchmark configuration) and *NFP values* is a space-separate series of *NFP=value* entries (i.e., benchmark output).
All measurements of a given *Key* *Attribute* combination must use the same set of NFP names.
Parameter names may be different -- parameters that are present in other lines of the same *Key* *Attribute* will be treated as undefined in those lines where they are missing.

Use `bin/analyze-log.py file1.txt file2.txt ...` for analysis.

## Model Types

dfatool supports six types of performance models:

* CART: Regression Trees
* DECART: Regression Trees with exclusively binary features/parameters
* XGB: Regression Forests
* LMT: Linear Model Trees
* RMT: Regression Model Trees
* Least-Squares Regression

Least-Squares Regression is essentially a subset of RMT with just a single tree node.
LMT and RMT differ significantly, as LMT uses a learning algorithm that starts out with a DECART and uses bottom-up pruning to turn it into an LMT, whereas RMT build a DECART that only considers parameters that are not suitable for least-squares regression and then uses least-squares regression to find and fit leaf functions.

By default, dfatool uses heuristics to determine whether it should generate a simple least-squares regression function or a fully-fledge RMT.
Use arguments (e.g. `--force-tree`) and environment variables (see below) to change which kinds of models it considers.

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
