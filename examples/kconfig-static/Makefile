default:
	@true

config:
	kconfig-qconf Kconfig

randconfig:
	kconfig-conf --randconfig Kconfig

clean:
	@true

nfpvalues:
	@./nfpvalues.sh

nfpkeys:
	@cat nfpkeys.json

.PHONY: default config randconfig clean nfpvalues nfpkeys
