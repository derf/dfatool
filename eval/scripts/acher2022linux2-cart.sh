#!/bin/sh

DFATOOL_MODEL=cart DFATOOL_CSV_IGNORE=active_options DFATOOL_CSV_OBSERVATIONS=perf \
	exec ../bin/analyze-log.py \
	--force-tree --skip-param-stats \
	~/var/ess/papers/splc-rmt/eval/acher-linux-zenodo/Linux.csv.xz \
	--export-dref out/acher2022linux2-cart.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
