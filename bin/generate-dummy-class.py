#!/usr/bin/env python3

from dfatool.aspectc import Repo
from dfatool.codegen import MultipassDriver
from dfatool.automata import PTA
import yaml

with open("../multipass/model/driver/nrf24l01.dfa", "r") as f:
    driver_definition = yaml.safe_load(f)
    pta = PTA.from_yaml(driver_definition)
repo = Repo("../multipass/build/repo.acp")

enum = dict()

if "dummygen" in driver_definition and "enum" in driver_definition["dummygen"]:
    enum = driver_definition["dummygen"]["enum"]

drv = MultipassDriver("Nrf24l01", pta, repo.class_by_name["Nrf24l01"], enum=enum)

with open("../multipass/src/driver/dummy.cc", "w") as f:
    f.write(drv.impl)
with open("../multipass/include/driver/dummy.h", "w") as f:
    f.write(drv.header)
