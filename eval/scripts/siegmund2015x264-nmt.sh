#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	DFATOOL_CSV_OBSERVATIONS=Energy,PSNR,SSIM,Size,Speed,Time,Watt \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/x264.csv.xz \
	--export-dref out/siegmund2015x264-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
