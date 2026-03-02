#!/bin/sh

DFATOOL_CSV_IGNORE=active_options DFATOOL_CSV_OBSERVATIONS=perf \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/acher-linux-zenodo/Linux.csv.xz \
	--export-dref out/acher2022linux2-rmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
