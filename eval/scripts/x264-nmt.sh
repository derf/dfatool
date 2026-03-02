#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	exec ../bin/analyze-kconfig.py \
	--force-tree \
	~/var/ess/thesis/eval/input/x264.kconfig ~/var/ess/thesis/eval/input/x264enum.json.xz \
	--export-dref out/x264-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
