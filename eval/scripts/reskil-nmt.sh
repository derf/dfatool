#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	exec ../bin/analyze-kconfig.py \
	~/var/ess/thesis/eval/input/reskil.kconfig ~/var/ess/thesis/eval/input/reskil.json.xz \
	--export-dref out/reskil-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
