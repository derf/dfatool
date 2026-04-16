#!/bin/bash

oldpwd="$(dirname "$(realpath "$0")")"

run_cart() {
	export "$@"
	set -e
	multipass=$(mktemp -d)
	rsync -a ~/var/projects/multipass/ ${multipass}/
	bin/eval-model-codegen.py --multipass-base ${multipass} --verify --model CART --dataset-load "/tmp/regression-${n_features}.json.xz" --type ${type}
	rm -rf ${multipass}
}

run_xgb() {
	export "$@"

	# XGB uses multiple threads for training and inference.
	# Empirically, with more than four threads, the synchronization overhead is so high that it actually slows down the application.
	export OMP_NUM_THREADS=4

	set -e
	multipass=$(mktemp -d)
	rsync -a ~/var/projects/multipass/ ${multipass}/
	bin/eval-model-codegen.py --multipass-base ${multipass} --verify --model XGB --dataset-load "/tmp/regression-${n_features}.json.xz" --type ${type}
	rm -rf ${multipass}
}

export -f run_cart run_xgb

parallel --eta --header : \
	bin/eval-model-codegen.py --dataset-n-features {n_features} --dataset-n-samples 10000 --dataset-save /tmp/regression-{n_features}.json.xz \
	::: n_features $(seq 2 6)

echo
echo CART
echo

parallel --eta --joblog /tmp/cart.joblog --header : \
	run_cart DFATOOL_CART_MAX_DEPTH={max_depth} n_features={n_features} type={type} \
	::: max_depth 4 6 8 10 12 14 16 \
	::: n_features 3 4 5 \
	::: type int8_t int16_t int32_t float double

echo
echo XGB
echo

parallel --eta --joblog /tmp/xgb.joblog --header : \
	run_xgb DFATOOL_XGB_MAX_DEPTH={max_depth} DFATOOL_XGB_N_ESTIMATORS={n_estimators} n_features={n_features} type={type} \
	::: max_depth 4 6 8 10 12 \
	::: n_estimators 1 5 10 15 20 25 30 40 50 60 70 80 90 100 120 140 160 180 200 \
	::: n_features 3 4 5 \
	::: type int8_t int16_t int32_t float double
