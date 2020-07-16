#!/usr/bin/env python3

import lzma
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import sys
import xml.etree.ElementTree as ET


def plot_y(Y, **kwargs):
    plot_xy(np.arange(len(Y)), Y, **kwargs)


def plot_xy(X, Y, xlabel=None, ylabel=None, title=None, output=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    if title != None:
        fig.canvas.set_window_title(title)
    if xlabel != None:
        ax1.set_xlabel(xlabel)
    if ylabel != None:
        ax1.set_ylabel(ylabel)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)
    plt.plot(X, Y, "bo", markersize=2)
    if output:
        plt.savefig(output)
        with open("{}.txt".format(output), "w") as f:
            print("X Y", file=f)
            for i in range(len(X)):
                print("{} {}".format(X[i], Y[i]), file=f)
    else:
        plt.show()


filename = sys.argv[1]

with open(filename, "rb") as logfile:
    lines = []
    line = ""

    if ".xz" in filename:
        f = lzma.open(logfile)
    else:
        f = logfile

    while line != "</dlog>\n":
        line = f.readline().decode()
        lines.append(line)
    xml_header = "".join(lines)
    raw_header = f.read(8)
    data_offset = f.tell()
    raw_data = f.read()

    xml_header = xml_header.replace("1ua>", "X1ua>")
    xml_header = xml_header.replace("2ua>", "X2ua>")
    dlog = ET.fromstring(xml_header)
    channels = []
    for channel in dlog.findall("channel"):
        channel_id = int(channel.get("id"))
        sense_curr = channel.find("sense_curr").text
        sense_volt = channel.find("sense_volt").text
        model = channel.find("ident").find("model").text
        if sense_volt == "1":
            channels.append((channel_id, model, "V"))
        if sense_curr == "1":
            channels.append((channel_id, model, "A"))

    num_channels = len(channels)
    duration = int(dlog.find("frame").find("time").text)
    interval = float(dlog.find("frame").find("tint").text)
    real_duration = interval * int(len(raw_data) / (4 * num_channels))

    data = np.ndarray(
        shape=(num_channels, int(len(raw_data) / (4 * num_channels))), dtype=np.float32
    )

    iterator = struct.iter_unpack(">f", raw_data)
    channel_offset = 0
    measurement_offset = 0
    for value in iterator:
        data[channel_offset, measurement_offset] = value[0]
        if channel_offset + 1 == num_channels:
            channel_offset = 0
            measurement_offset += 1
        else:
            channel_offset += 1

if int(real_duration) != duration:
    print(
        "Measurement duration: {:f} of {:d} seconds at {:f} µs per sample".format(
            real_duration, duration, interval * 1000000
        )
    )
else:
    print(
        "Measurement duration: {:d} seconds at {:f} µs per sample".format(
            duration, interval * 1000000
        )
    )

for i, channel in enumerate(channels):
    channel_id, channel_model, channel_type = channel
    print(
        "channel {:d} ({:s}): min {:f}, max {:f}, mean {:f} {:s}".format(
            channel_id,
            channel_model,
            np.min(data[i]),
            np.max(data[i]),
            np.mean(data[i]),
            channel_type,
        )
    )

    if (
        i > 0
        and channel_type == "A"
        and channels[i - 1][2] == "V"
        and channel_id == channels[i - 1][0]
    ):
        power = data[i - 1] * data[i]
        power = 3.6 * data[i]
        print(
            "channel {:d} ({:s}): min {:f}, max {:f}, mean {:f} W".format(
                channel_id, channel_model, np.min(power), np.max(power), np.mean(power)
            )
        )
        min_power = np.min(power)
        max_power = np.max(power)
        power_border = np.mean([min_power, max_power])
        low_power = power[power < power_border]
        high_power = power[power >= power_border]
        plot_y(power)
        print(
            "    avg low / high power (delta): {:f} / {:f} ({:f}) W".format(
                np.mean(low_power),
                np.mean(high_power),
                np.mean(high_power) - np.mean(low_power),
            )
        )
        # plot_y(low_power)
        # plot_y(high_power)
        high_power_durations = []
        current_high_power_duration = 0
        for is_hpe in power >= power_border:
            if is_hpe:
                current_high_power_duration += interval
            else:
                if current_high_power_duration > 0:
                    high_power_durations.append(current_high_power_duration)
                current_high_power_duration = 0
        print(
            "    avg high-power duration: {:f} µs".format(
                np.mean(high_power_durations) * 1000000
            )
        )

# print(xml_header)
# print(raw_header)
# print(channels)
# print(data)
# print(np.mean(data[0]))
# print(np.mean(data[1]))
# print(np.mean(data[0] * data[1]))
