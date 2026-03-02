#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	DFATOOL_CSV_OBSERVATIONS='Measured Value' \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/javagc.csv.xz \
	--export-dref out/siegmund2015javagc-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
