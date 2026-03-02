#!/bin/sh

DFATOOL_MODEL=cart DFATOOL_CSV_OBSERVATIONS=Performance \
	exec ../bin/analyze-log.py \
	--skip-param-stats \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/hipacc.csv \
	--export-dref out/siegmund2015hipacc-cart.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
