#!/bin/sh

DFATOOL_CSV_IGNORE=id DFATOOL_CSV_OBSERVATIONS=vmlinux \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/acher-linux-kaggle/train.csv.xz \
	--export-dref out/acher2022linux1-rmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
