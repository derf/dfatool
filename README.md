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
| `DFATOOL_KCONF_WITH_CHOICE_NODES` | 0, **1** | Generate enum parameters from kconfig choice nodes; ignore corresponding boolean config options. |
| `DFATOOL_KCONF_IGNORE_NUMERIC` | **0**, 1 | Ignore numeric (int/hex) configuration options. Useful for comparison with CART/DECART. |
| `DFATOOL_KCONF_IGNORE_STRING` | **0**, 1 | Ignore string configuration options. Useful for comparison with CART/DECART. |
| `DFATOOL_FIT_LINEAR_ONLY` | **0**, 1 | Only consider linear functions (a + bx) in regression analysis. Useful for comparison with Linear Model Trees / M5. |
| `DFATOOL_REGRESSION_SAFE_FUNCTIONS` | **0**, 1 | Use safe functions only (e.g. 1/x returnning 1 for x==0) |
| `DFATOOL_DTREE_NONBINARY_NODES` | 0, **1** | Enable non-binary nodes (i.e., nodes with more than two children corresponding to enum variables) in decision trees |
| `DFATOOL_DTREE_LOSS_IGNORE_SCALAR` | **0**, 1 | Ignore scalar parameters when computing the loss for split node candidates. Instead of computing the loss of a single partition for each `x_i == j`, compute the loss of partitions for `x_i == j` in which non-scalar parameters vary and scalar parameters are constant. This way, scalar parameters do not affect the decision about which non-scalar parameter to use for splitting. |
