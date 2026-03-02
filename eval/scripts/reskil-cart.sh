#!/bin/sh

DFATOOL_MODEL=cart DFATOOL_KCONF_WITH_CHOICE_NODES=0 DFATOOL_PARAM_CATEGORICAL_TO_SCALAR=1 \
	exec ../bin/analyze-kconfig.py \
	--param-shift *=none-to-0 --force-tree --skip-param-stats \
	~/var/ess/thesis/eval/input/reskil.kconfig ~/var/ess/thesis/eval/input/reskilboolean.json.xz \
	--export-dref out/reskil-cart.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
