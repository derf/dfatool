#!/bin/sh

. ./.config

X=0
Y=0

if [ "$CONFIG_X5" = y ]; then
	X=5
fi
if [ "$CONFIG_X6" = y ]; then
	X=6
fi

if [ "$CONFIG_Y0" = y ]; then
	Y=0
fi
if [ "$CONFIG_Y4" = y ]; then
	Y=4
fi

echo '{"Synthetic": {"X": '$X', "Y": '$Y'}}'
