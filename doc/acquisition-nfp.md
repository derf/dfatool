# NFP Benchmarks for SPLs

[explore-kconfig.py](bin/explore-kconfig.py) works with any product line that supports Kconfig and is capable of describing the non-functional properties of individual products.
To do so, the product line's build system must provide the **make**, **make clean**, **make randconfig**, **make nfpvalues** and **make nfpkeys** commands.
**make nfpvalues** is expected to print a JSON dict describing the non-functional property values of the current build;
**make nfpkeys** is expected to print a JSON dict with meta-data about those.
All of these commands can be changed, see `bin/explore-kconfig.py --help`.

See [explore-and-model-static](examples/explore-and-model-static) for a simple example project, and [busybox.sh](examples/busybox.sh) for a more complex one.

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
Most approaches in the literature use boolean variables.
Note that, when working with exported observations, **DFATOOL_KCONF_WITH_CHOICE_NODES** must have the same value in the `--export-observations` call and in subsequent `analyze-kconfig.py` calls using these observations.
