#!/bin/sh

exec ../bin/analyze-kconfig.py \
	~/var/ess/thesis/eval/input/reskil.kconfig ~/var/ess/thesis/eval/input/reskil.json.xz \
	--export-dref out/reskil-rmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
