#!/bin/sh

DFATOOL_RMT_FUNCTION_LEAVES=0 DFATOOL_RMT_PRUNE=1 \
	DFATOOL_CSV_OBSERVATIONS=compile-avgmem,compile-cpu,compile-exit,compile-ioin,compile-ioout,compile-maxmem,compile-power,compile-real,compile-size,compile-sys,compile-user,run-avgmem,run-cpu,run-exit,run-ioin,run-ioout,run-maxmem,run-power,run-real,run-sys,run-user \
	exec ../bin/analyze-log.py \
	--skip-param-stats --force-tree \
	~/var/ess/papers/splc-rmt/eval/siegmund2015esecfse/sac.csv.xz \
	--export-dref out/siegmund2015sac-nmt.tex \
	--cross-validate=kfold:10 --parameter-aware-cross-validation \
	--progress \
	--show-model-error --show-model-complexity
