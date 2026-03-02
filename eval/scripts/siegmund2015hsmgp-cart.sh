#!/bin/sh

DFATOOL_MODEL=cart DFATOOL_CSV_OBSERVATIONS=AverageTimePerIteration \
	exec ../bin/analyze-log.py \
	--skip-param-stats \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/hsmgp.csv \
	--export-dref out/siegmund2015hsmgp-cart.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
