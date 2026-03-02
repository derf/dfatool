#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	DFATOOL_CSV_OBSERVATIONS=AverageTimePerIteration \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/hsmgp.csv \
	--export-dref out/siegmund2015hsmgp-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
