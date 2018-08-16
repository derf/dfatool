#!/usr/bin/env python3

import lzma
import numpy as np
import os
import struct
import sys
import xml.etree.ElementTree as ET

filename = sys.argv[1]

with open(filename, 'rb') as logfile:
    lines = []
    line = ''

    if '.xz' in filename:
        f = lzma.open(logfile)
    else:
        f = logfile

    while line != '</dlog>\n':
        line = f.readline().decode()
        lines.append(line)
    xml_header = ''.join(lines)
    raw_header = f.read(8)
    data_offset = f.tell()
    raw_data = f.read()

    xml_header = xml_header.replace('1ua>', 'X1ua>')
    xml_header = xml_header.replace('2ua>', 'X2ua>')
    dlog = ET.fromstring(xml_header)
    channels = []
    for channel in dlog.findall('channel'):
        channel_id = int(channel.get('id'))
        sense_curr = channel.find('sense_curr').text
        sense_volt = channel.find('sense_volt').text
        model = channel.find('ident').find('model').text
        if sense_volt == '1':
            channels.append((channel_id, model, 'V'))
        if sense_curr == '1':
            channels.append((channel_id, model, 'A'))

    num_channels = len(channels)
    duration = int(dlog.find('frame').find('time').text)
    interval = float(dlog.find('frame').find('tint').text)
    real_duration = interval * int(len(raw_data) / (4 * num_channels))

    data = np.ndarray(shape=(num_channels, int(len(raw_data) / (4 * num_channels))), dtype=np.float32)

    iterator = struct.iter_unpack('>f', raw_data)
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
    print('Measurement duration: {:f} of {:d} seconds at {:f} µs per sample'.format(
        real_duration, duration, interval * 1000000))
else:
    print('Measurement duration: {:d} seconds at {:f} µs per sample'.format(
        duration, interval * 1000000))

for i, channel in enumerate(channels):
    channel_id, channel_model, channel_type = channel
    print('channel {:d} ({:s}): min {:f}, max {:f}, mean {:f} {:s}'.format(
        channel_id, channel_model, np.min(data[i]), np.max(data[i]), np.mean(data[i]),
        channel_type))

    if i > 0 and channel_type == 'A' and channels[i-1][2] == 'V' and channel_id == channels[i-1][0]:
        power = data[i-1] * data[i]
        print('channel {:d} ({:s}): min {:f}, max {:f}, mean {:f} W'.format(
            channel_id, channel_model, np.min(power), np.max(power), np.mean(power)))

#print(xml_header)
#print(raw_header)
#print(channels)
#print(data)
#print(np.mean(data[0]))
#print(np.mean(data[1]))
#print(np.mean(data[0] * data[1]))
