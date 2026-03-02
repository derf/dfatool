#!/bin/sh

DFATOOL_MODEL=cart DFATOOL_KCONF_WITH_CHOICE_NODES=0 DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1 \
	exec ../bin/analyze-kconfig.py \
	--param-shift '*=none-to-0' --force-tree --skip-param-stats \
	~/var/ess/thesis/eval/input/x264.kconfig ~/var/ess/thesis/eval/input/x264boolean.json.xz \
	--export-dref out/x264-cart.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
