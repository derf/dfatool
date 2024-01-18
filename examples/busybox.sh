#!/bin/sh

set -ex

wget https://busybox.net/downloads/busybox-1.35.0.tar.bz2
tar xjf busybox-1.35.0.tar.bz2
rm busybox-1.35.0.tar.bz2

# Generate Config.in files
( cd busybox-1.35.0 ; make gen_build_files )

git clone --recursive https://ess.cs.uos.de/git/software/dfatool.git

dfatool/bin/kconfig-expand-includes busybox-1.35.0/Config.in > busybox-1.35.0/Kconfig

cat > busybox-1.35.0/nfpkeys.json <<__EOF__
{"ELF": {"Size": {"unit": "B", "description": "Binary Size", "minimize": true}, "RAM": {"unit": "B", "description": "static RAM", "minimize": true}}}
__EOF__

cat > busybox-1.35.0/Makefile.local << __EOF__
.PHONY: nfpkeys
nfpkeys:
	@cat nfpkeys.json

.PHONY: nfpvalues
nfpvalues:
	@scripts/nfpvalues.py size _ data,bss

__EOF__

cat > busybox-1.35.0/scripts/nfpvalues.py << __EOF__
#!/usr/bin/env python3

import json
import re
import subprocess
import sys


def main(size_executable, rom_sections, ram_sections):
    rom_sections = rom_sections.split(",")
    ram_sections = ram_sections.split(",")

    status = subprocess.run(
        [size_executable, "-A", "busybox"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    section_size = dict()

    for line in status.stdout.split("\n"):
        match = re.match("[.](\S+)\s+(\d+)", line)
        if match:
            section = match.group(1)
            size = int(match.group(2))
            section_size[section] = size

    status = subprocess.run(
        ["stat", "-c", "%s", "busybox"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    binary_size = int(status.stdout)

    total = {
        "Size": binary_size,
        "RAM": sum(map(lambda section: section_size[section], ram_sections)),
    }

    output = {"ELF": total}

    print(json.dumps(output))


if __name__ == "__main__":
    main(*sys.argv[1:])

__EOF__

chmod 755 busybox-1.35.0/scripts/nfpvalues.py

set +ex

cat <<__EOF__

busybox has been prepared for dfatool data acquisition for NFP model generation.

Example usage with random sampling (10,000 samples)

> mkdir data
> cd data
> ../dfatool/bin/explore-kconfig.py --random 10000 ../busybox-1.35.0

Alternatively, you can run multiple build processes in parallel.

> for i in {1..10}; do rsync -a busybox-1.35.0/ busybox-build-\${i}/ ; done
> mkdir data
> cd data
> for i in {1..10}; do ../dfatool/bin/explore-kconfig.py --random 1000 ../busybox-build-\${i} & done

Once everything is done, you have various options for generating a kconfig-webconf performance model.
To do so, call bin/analyze-kconfig.py from the data directory.
For example, to generate a CART:

> DFATOOL_DTREE_SKLEARN_CART=1 DFATOOL_PARAM_CATEGORIAL_TO_SCALAR=1 DFATOOL_KCONF_WITH_CHOICE_NODES=0 ~/var/ess/aemr/dfatool/bin/analyze-kconfig.py --force-tree --skip-param-stats --export-webconf /tmp/busybox-cart.json ../busybox-1.35.0/Kconfig .

By adding the option "--cross-validation kfold:10", you can determine the model prediction error.

__EOF__
