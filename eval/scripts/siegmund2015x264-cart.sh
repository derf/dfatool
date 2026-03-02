#!/bin/sh

DFATOOL_MODEL=cart DFATOOL_CSV_OBSERVATIONS='Energy,PSNR,SSIM,Size,Speed,Time,Watt' \
	exec ../bin/analyze-log.py \
	--skip-param-stats \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/x264.csv.xz \
	--export-dref out/siegmund2015x264-cart.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
