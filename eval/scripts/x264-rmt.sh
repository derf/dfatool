#!/bin/sh

exec ../bin/analyze-kconfig.py \
	--force-tree \
	~/var/ess/thesis/eval/input/x264.kconfig ~/var/ess/thesis/eval/input/x264enum.json.xz \
	--export-dref out/x264-rmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
