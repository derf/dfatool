#!/bin/sh

DFATOOL_CSV_OBSERVATIONS=AverageTimePerIteration \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/hsmgp.csv \
	--export-dref out/siegmund2015hsmgp-rmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
