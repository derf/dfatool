#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	DFATOOL_ULS_SKIP_CODEPENDENT_CHECK=1 \
	exec ../bin/analyze-kconfig.py \
	--force-tree --skip-param-stats \
	~/var/ess/thesis/eval/input/busybox-1.35.0.kconfig ~/var/ess/thesis/eval/input/busybox-1.35.0-randconfig-bool.json.xz \
	--export-dref out/busybox-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
