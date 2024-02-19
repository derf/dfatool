# NFP Models for SPLs

[analyze-kconfig.py](bin/analyze-kconfig.py) builds, evaluates, and exports NFP models from explore-kconfig measurements.
Command-line options and environment variables determine which kind of NFP model it generates.

For example, when called in the benchmark data directory from the previous section, the following command generates a CART model for busybox and stores it in a kconfig-webconf-compatible format in `busybox.json`.
Classification and Regression Trees (CART) are capable of generating accurate models from a relatively small amount of samples, but only annotate important features.
Hence, after loading a CART model into kconfig-webconf, only a small subset of busybox features will be annotated with NFP deltas.

```
DFATOOL_DTREE_SKLEARN_CART=1 DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1 DFATOOL_KCONF_WITH_CHOICE_NODES=0 .../dfatool/bin/analyze-kconfig.py --export-webconf busybox.json --force-tree ../busybox-1.35.0/Kconfig .
```

Refer to the [kconfig-webconf README](https://ess.cs.uos.de/git/software/kconfig-webconf/-/blob/master/README.md#user-content-performance-aware-configuration) for details on using the generated model.

We also have a short [video example](https://ess.cs.uos.de/static/videos/splc22-kconfig-webconf.mp4) illustrating this workflow.
