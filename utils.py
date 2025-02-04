import struct
import re
import numpy as np
from BCG_preprocessing import signal_quality_assessment_v4
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks, butter, lfilter, filtfilt
from datetime import datetime
import pytz

def convert_unix_to_newyork_time(unix_timestamp):
    newyork_tz = pytz.timezone('America/New_York')
    utc_time = datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=pytz.utc)
    newyork_time = utc_time.astimezone(newyork_tz)
    return newyork_time.strftime('%Y-%m-%d %H:%M:%S')


def band_pass_filter(data, Fs, low, high, order, causal = False):
    b, a = butter(order, [low/(Fs * 0.5), high/(Fs * 0.5)], 'bandpass')
    # perform band pass filter
    if causal:
        filtered_data = lfilter(b, a, data)
    else:
        filtered_data = filtfilt(b, a, data)
    return filtered_data


def get_hr_from_abp(abp):
    x = abp
    x = (x - np.mean(x)) / np.std(x)
    denoised_sig = band_pass_filter(data=x, Fs=128,
                                    low=1, high=20, order=5,
                                    causal=False)
    acf_x = acf(denoised_sig, nlags=len(denoised_sig))
    peaks = find_peaks(acf_x, height=np.max(acf_x)*0.5)[0]
    if len(peaks)==0:
        return -20
    if peaks[0]<=10:
        if len(peaks)==1:
            return -20
        else:
            peaks = peaks[1:]
    hr = 60.0*128 / peaks[0]
    return hr

def parse_beddot_data(msg):
    bytedata=msg.payload
    mac_addr = ":".join(f"{x:02x}" for x in struct.unpack("!BBBBBB", bytedata[0:6]))
    data_len =struct.unpack("H",bytedata[6:8])[0]
    timestamp=struct.unpack("L",bytedata[8:16])[0]  # in micro second
    data_interval=struct.unpack("I",bytedata[16:20])[0]  # in micro second
    signal_type = struct.unpack("i", bytedata[20:24])[0]

    timestamp -=data_interval
    # data=[0]*int((len(bytedata)-20)/4)
    data=[0]*data_len
    index=24
    for i in range(data_len):
        if len(bytedata[index:index+4]) == 4:
            data[i] = struct.unpack("i", bytedata[index:index+4])[0]
            index +=4
            # timestamp +=data_interval
    # print("mac_addr,",mac_addr,timestamp,data_interval)
    if (data_interval < 10**6):     # less than 1 second
        timestamp = (timestamp // data_interval)*data_interval*1000 #align timestamp and convert to nano second.
    else:
        timestamp = (timestamp // 1000000)* 10**9   # align to second
    # timestamp = (timestamp // 10000)* 10**7 #convert to nano second. resolution=10ms
    data_interval *=1000 #convert to nano second

    return  mac_addr, timestamp, data_interval, signal_type, data

def encode_beddot_data(mac_addr, timestamp, data_interval, data):
    mac_bytes = bytes(int(x, 16) for x in mac_addr.split(":")) # mac address should be in the form of "11:22:33:44:55:66"
    data_len = len(data) # data length is always 100, which means one second
    data_len_bytes = struct.pack("H", data_len)
    timestamp_bytes = struct.pack("L", int(timestamp * 1e6)) # timestamp is in second (orignal is in us)
    data_interval_bytes = struct.pack("I", data_interval) # data interval is in us
    # signal_type_bytes = struct.pack("i", signal_type) # 0 is BSG, 1 is ECG
    data_bytes = b''.join(struct.pack("i", int(d)) for d in data)
    encoded_data = mac_bytes + data_len_bytes + timestamp_bytes + data_interval_bytes + data_bytes
    return encoded_data


def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        # print('invalid mac address, use 0 instead')
        return int(0)
    return int(res.group(0).replace(':', ''), 16)

def int_to_mac(mac_int):
    mac_int = int(mac_int)
    hex_str = f"{mac_int:012x}"
    mac = ':'.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
    return mac

def quality_control(bsg, fs):
    res = signal_quality_assessment_v4(x = bsg, Fs = fs,
                                        n_lag = len(bsg)//2,
                                        low = 1, high = 20,
                                        acf_low = int(0.25 * fs),
                                        acf_high = int(1.2 * fs),
                                        f_low = 0.6)
    if not res[0]:
        res = signal_quality_assessment_v4(x = bsg, Fs = fs,
                                        n_lag = len(bsg)//2,
                                        low = 1, high = 10,
                                        acf_low = int(0.25 * fs),
                                        acf_high = int(1.2 * fs),
                                        f_low = 0.6, diff = 0)
    return res[0], res[-1]