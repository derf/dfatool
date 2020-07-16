#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from dfatool.loader import MIMOSA, KeysightCSV
from dfatool.utils import running_mean

voltage = float(sys.argv[1])
shunt = float(sys.argv[2])

mimfile = "../data/20161114_arb_%d.mim" % shunt
csvfile = "../data/20161114_%d_arb.csv" % shunt

mim = MIMOSA(voltage, shunt)
ks = KeysightCSV()

charges, triggers = mim.load_data(mimfile)
timestamps, currents = ks.load_data(csvfile)


def calfunc330(charge):
    if charge < 140.210488888889:
        return 0
    if charge <= 526.507377777778:
        return float(charge) * 0.0941215500652876 + -13.196828549634
    else:
        return float(charge) * 0.0897304193584184 + -47.2437278033012 + 36.358862


def calfunc82(charge):
    if charge < 126.993600:
        return 0
    if charge <= 245.464889:
        return charge * 0.306900 + -38.974361
    else:
        return charge * 0.356383 + -87.479495 + 36.358862


def calfunc33(charge):
    if charge < 127.000000:
        return 0
    if charge <= 127.211911:
        return charge * 171.576006 + -21790.152700
    else:
        return charge * 0.884357 + -112.500777 + 36.358862


calfuncs = {
    33: calfunc33,
    82: calfunc82,
    330: calfunc330,
}

vcalfunc = np.vectorize(calfuncs[int(shunt)], otypes=[np.float64])

# plt.plot(np.arange(0, 1000, 0.01), vcalfunc(np.arange(0, 1000, 0.01)))
# plt.xlabel('Rohdatenwert')
# plt.ylabel('Strom [µA]')
# plt.show()
# sys.exit(0)

mim_x = np.arange(len(charges) - 199) * 1e-5
mim_y = running_mean(mim.charge_to_current_nocal(charges), 200) * 1e-6
cal_y = running_mean(vcalfunc(charges), 200) * 1e-6
ks_x = timestamps[: len(timestamps) - 9]
ks_y = running_mean(currents, 10)

# look for synchronization opportunity in first 5 seconds
mim_sync_idx = 0
ks_sync_idx = 0
for i in range(0, 500000):
    if mim_sync_idx == 0 and mim_y[i] > 0.001:
        mim_sync_idx = i
    if ks_sync_idx == 0 and ks_y[i] > 0.001:
        ks_sync_idx = i

mim_x = mim_x - mim_x[mim_sync_idx]
ks_x = ks_x - ks_x[ks_sync_idx]

mim_max_start = int(len(mim_y) * 0.4)
mim_max_end = int(len(mim_y) * 0.6)
mim_start_end = int(len(mim_y) * 0.1)
mim_end_start = int(len(mim_y) * 0.9)
mim_max = np.max(mim_y[mim_max_start:mim_max_end])
mim_min1 = np.min(mim_y[:mim_start_end])
mim_min2 = np.min(mim_y[mim_end_start:])
mim_center = 0
mim_start = 0
mim_end = 0
for i, y in enumerate(mim_y):
    if y == mim_max and i / len(mim_y) > 0.4 and i / len(mim_y) < 0.6:
        mim_center = i
    elif y == mim_min1 and i / len(mim_y) < 0.1:
        mim_start = i
    elif y == mim_min2 and i / len(mim_y) > 0.9:
        mim_end = i

plt.plot([mim_x[mim_center]], [mim_y[mim_center]], "yo")
plt.plot([mim_x[mim_start]], [mim_y[mim_start]], "yo")
plt.plot([mim_x[mim_end]], [mim_y[mim_end]], "yo")
#
(mimhandle,) = plt.plot(mim_x, mim_y, "r-", label="MIMOSA")
# calhandle, = plt.plot(mim_x, cal_y, "g-", label='MIMOSA (autocal)')
(kshandle,) = plt.plot(ks_x, ks_y, "b-", label="Keysight")
# plt.legend(handles=[mimhandle, calhandle, kshandle])
plt.xlabel("Zeit [s]")
plt.ylabel("Strom [A]")
plt.grid(True)

ks_steps_up = []
ks_steps_down = []
mim_steps_up = []
mim_steps_down = []

skip = 0
for i, gradient in enumerate(np.gradient(ks_y, 10000)):
    if (
        gradient > 0.5e-9
        and i - skip > 200
        and ks_x[i] < mim_x[mim_center]
        and ks_x[i] > 5
    ):
        plt.plot([ks_x[i]], [ks_y[i]], "go")
        ks_steps_up.append(i)
        skip = i
    elif (
        gradient < -0.5e-9
        and i - skip > 200
        and ks_x[i] > mim_x[mim_center]
        and ks_x[i] < mim_x[mim_end]
    ):
        plt.plot([ks_x[i]], [ks_y[i]], "g*")
        ks_steps_down.append(i)
        skip = i

j = 0
for i, ts in enumerate(mim_x):
    if j < len(ks_steps_up) and ts > ks_x[ks_steps_up[j]]:
        mim_steps_up.append(i)
        j += 1

j = 0
for i, ts in enumerate(mim_x):
    if j < len(ks_steps_down) and ts > ks_x[ks_steps_down[j]]:
        mim_steps_down.append(i)
        j += 1

print(ks_steps_up)
print(mim_steps_up)

mim_values = []
cal_values = []
ks_values = []

for i in range(1, len(ks_steps_up)):
    mim_values.append(np.mean(mim_y[mim_steps_up[i - 1] : mim_steps_up[i]]))
    cal_values.append(np.mean(cal_y[mim_steps_up[i - 1] : mim_steps_up[i]]))
    ks_values.append(np.mean(ks_y[ks_steps_up[i - 1] : ks_steps_up[i]]))
    print(
        "step %d avg %5.3f vs %5.3f vs %5.3f mA"
        % (
            i,
            np.mean(ks_y[ks_steps_up[i - 1] : ks_steps_up[i]]) * 1e3,
            np.mean(mim_y[mim_steps_up[i - 1] : mim_steps_up[i]]) * 1e3,
            np.mean(cal_y[mim_steps_up[i - 1] : mim_steps_up[i]]) * 1e3,
        )
    )
for i in range(1, len(ks_steps_down)):
    mim_values.append(np.mean(mim_y[mim_steps_down[i - 1] : mim_steps_down[i]]))
    cal_values.append(np.mean(cal_y[mim_steps_down[i - 1] : mim_steps_down[i]]))
    ks_values.append(np.mean(ks_y[ks_steps_down[i - 1] : ks_steps_down[i]]))
    print(
        "step %d avg %5.3f vs %5.3f vs %5.3f mA"
        % (
            i,
            np.mean(ks_y[ks_steps_down[i - 1] : ks_steps_down[i]]) * 1e3,
            np.mean(mim_y[mim_steps_down[i - 1] : mim_steps_down[i]]) * 1e3,
            np.mean(cal_y[mim_steps_down[i - 1] : mim_steps_down[i]]) * 1e3,
        )
    )

mim_values = np.array(mim_values)
cal_values = np.array(cal_values)
ks_values = np.array(ks_values)

plt.show()

plt.hist(
    ks_y[ks_steps_up[48] : ks_steps_up[49]] * 1e3,
    100,
    normed=0,
    facecolor="blue",
    alpha=0.8,
)
plt.xlabel("mA Keysight")
plt.ylabel("#")
plt.grid(True)
plt.show()
plt.hist(
    mim_y[mim_steps_up[48] : mim_steps_up[49]] * 1e3,
    100,
    normed=0,
    facecolor="blue",
    alpha=0.8,
)
plt.xlabel("mA MimosaGUI")
plt.ylabel("#")
plt.grid(True)
plt.show()

(mimhandle,) = plt.plot(
    ks_values * 1e3, mim_values * 1e3, "ro", label="Unkalibriert", markersize=4
)
(calhandle,) = plt.plot(
    ks_values * 1e3, cal_values * 1e3, "bs", label="Kalibriert", markersize=4
)
plt.legend(handles=[mimhandle, calhandle])
plt.xlabel("mA Keysight")
plt.ylabel("mA MIMOSA")
plt.grid(True)
plt.show()

(mimhandle,) = plt.plot(
    ks_values * 1e3,
    (mim_values - ks_values) * 1e3,
    "ro",
    label="Unkalibriert",
    markersize=4,
)
(calhandle,) = plt.plot(
    ks_values * 1e3,
    (cal_values - ks_values) * 1e3,
    "bs",
    label="Kalibriert",
    markersize=4,
)
plt.legend(handles=[mimhandle, calhandle])
plt.xlabel("Sollstrom [mA]")
plt.ylabel("Messfehler MIMOSA [mA]")
plt.grid(True)
plt.show()

(mimhandle,) = plt.plot(
    ks_values * 1e3, (mim_values - ks_values) * 1e3, "r--", label="Unkalibriert"
)
(calhandle,) = plt.plot(
    ks_values * 1e3, (cal_values - ks_values) * 1e3, "b-", label="Kalibriert"
)
plt.legend(handles=[mimhandle, calhandle])
plt.xlabel("Sollstrom [mA]")
plt.ylabel("Messfehler MIMOSA [mA]")
plt.grid(True)
plt.show()

(mimhandle,) = plt.plot(
    ks_values * 1e3,
    (mim_values - ks_values) / ks_values * 100,
    "ro",
    label="Unkalibriert",
    markersize=4,
)
(calhandle,) = plt.plot(
    ks_values * 1e3,
    (cal_values - ks_values) / ks_values * 100,
    "bs",
    label="Kalibriert",
    markersize=4,
)
plt.legend(handles=[mimhandle, calhandle])
plt.xlabel("Sollstrom [mA]")
plt.ylabel("Messfehler MIMOSA [%]")
plt.grid(True)
plt.show()

(mimhandle,) = plt.plot(
    ks_values * 1e3,
    (mim_values - ks_values) / ks_values * 100,
    "r--",
    label="Unkalibriert",
)
(calhandle,) = plt.plot(
    ks_values * 1e3,
    (cal_values - ks_values) / ks_values * 100,
    "b-",
    label="Kalibriert",
)
plt.legend(handles=[mimhandle, calhandle])
plt.xlabel("Sollstrom [mA]")
plt.ylabel("Messfehler MIMOSA [%]")
plt.grid(True)
plt.show()

# mimhandle, = plt.plot(mim_x, np.gradient(mim_y, 10000), "r-", label='MIMOSA')
# kshandle, = plt.plot(ks_x, np.gradient(ks_y, 10000), "b-", label='Keysight')
# plt.legend(handles=[mimhandle, kshandle])
# plt.xlabel('Zeit [s]')
# plt.ylabel('Strom [A]')
# plt.show()
