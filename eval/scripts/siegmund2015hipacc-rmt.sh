#!/bin/sh

DFATOOL_CSV_OBSERVATIONS=Performance \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/hipacc.csv \
	--export-dref out/siegmund2015hipacc-rmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
