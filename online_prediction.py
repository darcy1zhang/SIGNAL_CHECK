# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
import subprocess
import sys
import warnings
import os
import argparse
from time import sleep
import pickle
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import threading
import queue
import time
from dateutil.parser import isoparse
import pytz
# from common import *
import warnings
from urllib3.exceptions import InsecureRequestWarning
import operator


MAX_POINT_PER_WRITE = 5000

# Suppress the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)
msg_queue = queue.Queue(maxsize=2)

def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))]*2)])

def switch_UNX_to_NY_Time(time_int):
    utc_time = datetime.utcfromtimestamp(time_int)
    ny_timezone = pytz.timezone("America/New_York")
    ny_time = utc_time.replace(tzinfo=pytz.utc).astimezone(ny_timezone)
    return ny_time


def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
    # client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
    query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))
    # query = 'SELECT last("H") FROM "labelled" WHERE ("location" = \''+unit+'\')'
    result = influx.query(query)
    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    # print(times)
    # times = [local_time_epoch(item[:-1], "UTC") for item in times] # convert string time to epoch time
    data = values #np.array(values)
    return data

def compute_1_patient_correlation(influxdb_client, mac_address, start_time, end_time, duration=120, step=30, window_size=10):
    # start_epoch = int(start_time.timestamp())
    start_timestamp = start_time
    end_timestamp = start_timestamp + timedelta(seconds=duration)

    corr_list = []
    corr_mov_list = []
    time_list = []
    cnt = 0
    json_body = []
    while end_timestamp < end_time:
        #download HR labels
        HR_ = read_influx(influxdb_client, mac_address, 'icu_correct_label', 'HR', int(start_timestamp.timestamp()), int(end_timestamp.timestamp()))
        if len(HR_)<duration:
            start_timestamp = start_timestamp + timedelta(seconds=step)
            end_timestamp = start_timestamp + timedelta(seconds=duration)
            continue
        #download MAP labels
        Art_M_ = read_influx(influxdb_client, mac_address, 'icu_correct_label', 'Art_M', int(start_timestamp.timestamp()), int(end_timestamp.timestamp()))
        if len(Art_M_)<duration:
            start_timestamp = start_timestamp + timedelta(seconds=step)
            end_timestamp = start_timestamp + timedelta(seconds=duration)
            continue
        #compute correlation
        corr = np.corrcoef(HR_, Art_M_)[0,1]
        #compute correlation moving average
        HR_mov = np.convolve(HR_, np.ones(window_size) / window_size, mode='valid')
        Art_M_mov = np.convolve(Art_M_, np.ones(window_size) / window_size, mode='valid')
        corr_mov = np.corrcoef(HR_mov, Art_M_mov)[0,1]
        print(f'{start_timestamp}: Corr = {corr:.4f}, Moving Corr = {corr_mov:.4f}')
        #upload
        corr_list.append(corr)
        corr_mov_list.append(corr_mov)
        time_list.append(start_timestamp)
        cnt += 1
        json_point = {
            "measurement": 'Test',
            "tags": {'location': mac_address},  # Extract tags
            "time": start_timestamp.isoformat(timespec='microseconds'),
            "fields": {f'correlation_{duration}s': float(corr), f'correlation_moving_{duration}s': float(corr_mov)}#float(value_list[i])
            # Extract fields
        }
        # print(json_point)
        json_body.append(json_point)

        if (cnt % MAX_POINT_PER_WRITE) == 0:
            data_msg = json_body.copy()
            msg_queue.put(data_msg)
            json_body.clear()
            sleep(0.1)  # Adjust this value based on the CPU load on the server. Make sure the CPU load is below 70%
            print(f'Add {MAX_POINT_PER_WRITE} / {cnt}')


        start_timestamp = start_timestamp + timedelta(seconds=step)
        end_timestamp = start_timestamp + timedelta(seconds=duration)

    if json_body:
        msg_queue.put(json_body)
        print(f'Add {len(json_body)} at the end')
    
    
    return corr_list, corr_mov_list, time_list
    
def write_db(db_client):
    while True:
        data_body = msg_queue.get()
        try:
            db_client.write_points(data_body)
            print(f"{len(data_body)} Data Saved")
        except Exception as e:
            print(f"write_db: Failed to write data to the Influxdb")

        data_body.clear()

def connect_influxdb(influxdb_conf):
    verify = False
    influxdb_client = InfluxDBClient(influxdb_conf["ip"], influxdb_conf["port"], influxdb_conf["user"], influxdb_conf["password"],
                                     influxdb_conf["db"], ssl=influxdb_conf["ssl"], verify_ssl=verify)

    # Check the connection status
    try:
        if influxdb_client.ping():
            print(f"Successfully connected to the Influxdb!")
            # print("")
        else:
            print(f"Failed to connect to the Influxdb, exit")
            exit(1)
    except Exception as e:
        print(f"Ping failure, Connection to the Influxdb failed!, exit")
        exit(1)
    return influxdb_client

def check_folder_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def compute_energy(influxdb_client, mac_address, timestamp):
    start_timestamp = timestamp - timedelta(seconds=30)
    end_list = [2, 5, 10, 20, 30]
    eng = []
    
    end_timestamp = timestamp
    signal_ = read_influx(influxdb_client, mac_address, 'Z', 'value', int(start_timestamp.timestamp()), int(end_timestamp.timestamp()))
    if len(signal_)!=30*100:
        return None

    eng.append(np.sum(np.square(signal_[-2*100:]))/2)
    eng.append(np.sum(np.square(signal_[-5*100:]))/5)
    eng.append(np.sum(np.square(signal_[-10*100:]))/10)
    eng.append(np.sum(np.square(signal_[-20*100:]))/20)
    eng.append(np.sum(np.square(signal_[-30*100:]))/30)
  
  
    return eng

def initialize_csv(file_path, headers):
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

# 添加一行数据到CSV文件
def append_row(file_path, row_data, headers):
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writerow(row_data)

def upload_for_one_patient(influxdb_client, patient_ID, DL_path = './Orth/features/'):
    #read_result
    with open(os.path.join(DL_path, f'feature_{patient_ID}.pkl'), 'rb') as file:
        res_data = pickle.load(file)
    #id_save_dict = {'feature': test_fea, 'pred_label':test_pred_label, 'Y_test':Y_test, 'timestamp':time_test}
    res_timestamp = res_data['timestamp']
    res_Y_test = res_data['Y_test']
    
    cnt = 0
    json_body = []
    time_list = set(res_timestamp)

    #mac_address
    mac_address = int_to_mac(res_Y_test[0, 1])
    start_time = min(time_list)
    end_time = max(time_list)
    print(patient_ID, mac_address, switch_UNX_to_NY_Time(start_time), switch_UNX_to_NY_Time(end_time))

    cur_time = start_time
    while cur_time <= end_time:
        cur_timestamp = switch_UNX_to_NY_Time(cur_time)
        energy_list = compute_energy(influxdb_client, mac_address, cur_timestamp)
        cur_time = cur_time + 1
        
        if energy_list is None:
            continue
        field_dict = {
            'energy_2s': float(energy_list[0]), 
            'energy_5s': float(energy_list[1]), 
            'energy_10s': float(energy_list[2]), 
            'energy_20s': float(energy_list[3]), 
            'energy_30s': float(energy_list[4]), 
        }
        #Upload part
        cnt += 1
        #start_timestamp: datetime format
        json_point = {
            "measurement": 'Test',
            "tags": {'location': mac_address},  # Extract tags
            "time": cur_timestamp.isoformat(timespec='microseconds'),
            "fields": field_dict
            # Extract fields
        }
        json_body.append(json_point)

        if (cnt % MAX_POINT_PER_WRITE) == 0:
            data_msg = json_body.copy()
            msg_queue.put(data_msg)
            json_body.clear()
            sleep(0.1)  # Adjust this value based on the CPU load on the server. Make sure the CPU load is below 70%
            print(f'Add {MAX_POINT_PER_WRITE} / {cnt}')
            

    if json_body:
        msg_queue.put(json_body)
        print(f'Add {len(json_body)} to the end')
    
    print(f'Subject {patient_ID} has finished.')
    print()

    
if __name__ == '__main__':
    updest = {'ip': '172.22.112.251', 'port': 8086, 'db': 'icu', 'user': 'algtest', 'password': 'sensorweb711', 'ssl':False}
    updest_client = connect_influxdb(updest)
    write_db_thread = threading.Thread(target=write_db, args=(updest_client,))
    write_db_thread.daemon = True
    write_db_thread.start()

    downdest = {'ip': '172.22.112.251', 'port': 8086, 'db': 'shake', 'user': 'algtest', 'password': 'sensorweb711', 'ssl':False}
    downdest_client = connect_influxdb(downdest)

    DL_path = './Orth/features/'
    all_file_list = os.listdir(DL_path)

    for i in range(152):
        if f'feature_{i}.pkl' in all_file_list:
            upload_for_one_patient(influxdb_client=downdest_client, patient_ID=i, DL_path = './Orth/features/')
        else:
            continue

    downdest_client.close()
    updest_client.close()
    print('over')


# ========================================================================================

import numpy as np
import pytz
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random
import scipy.signal as signal
from statsmodels.tsa.stattools import acf
import csv
import pandas as pd
import subprocess
import sys
import warnings
import os
import argparse
from time import sleep
import pickle
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import threading
import queue
import time
from dateutil.parser import isoparse
from urllib3.exceptions import InsecureRequestWarning
import operator

MAX_POINT_PER_WRITE = 10000

# Suppress the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)
msg_queue = queue.Queue(maxsize=2)

def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))]*2)])

def switch_UNX_to_NY_Time(time_int):
    utc_time = datetime.utcfromtimestamp(time_int)
    ny_timezone = pytz.timezone("America/New_York")
    ny_time = utc_time.replace(tzinfo=pytz.utc).astimezone(ny_timezone)
    return ny_time

def write_db(db_client):
    while True:
        data_body = msg_queue.get()
        try:
            db_client.write_points(data_body)
            print(f"{len(data_body)} Data Saved")
        except Exception as e:
            print(f"write_db: Failed to write data to the Influxdb")

        data_body.clear()



def connect_influxdb(influxdb_conf):
    verify = False
    influxdb_client = InfluxDBClient(influxdb_conf["ip"], influxdb_conf["port"], influxdb_conf["user"], influxdb_conf["password"],
                                     influxdb_conf["db"], ssl=influxdb_conf["ssl"], verify_ssl=verify)

    # Check the connection status
    try:
        if influxdb_client.ping():
            print(f"Successfully connected to the Influxdb!")
            # print("")
        else:
            print(f"Failed to connect to the Influxdb, exit")
            exit(1)
    except Exception as e:
        print(f"Ping failure, Connection to the Influxdb failed!, exit")
        exit(1)
    return influxdb_client

def check_folder_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def band_pass_filter(data, Fs, low, high, order, causal = False):
    b, a = signal.butter(order, [low/(Fs * 0.5), high/(Fs * 0.5)], 'bandpass')
    # perform band pass filter
    if causal:
        filtered_data = signal.lfilter(b, a, data)
    else:
        filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def compute_hr(x):
    raw = x
    x = (x - np.mean(x)) / np.std(x)
    denoised_sig = band_pass_filter(data=x, Fs=128,
                                    low=1, high=20, order=5,
                                    causal=False)
    acf_x = acf(denoised_sig, nlags=len(denoised_sig))

    peaks = find_peaks(acf_x, height=np.max(acf_x)*0.5)[0]
    if len(peaks)==0:
        return None
    if peaks[0]<=10:
        if len(peaks)==1:
            return None
        else:
            peaks = peaks[1:]
    hr = 60.0*128 / peaks[0]


    # raw_peaks = find_peaks(raw, height=np.max(raw)*0.8, width=20)[0]
    # gap = raw_peaks[1:] - raw_peaks[:-1]
    # gap = 60.0*128 / np.median(gap)
    #visual
    return hr


def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
    # client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
    query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))
    # query = 'SELECT last("H") FROM "labelled" WHERE ("location" = \''+unit+'\')'
    result = influx.query(query)
    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    # print(times)
    # times = [local_time_epoch(item[:-1], "UTC") for item in times] # convert string time to epoch time
    data = values #np.array(values)
    return data


bsg_dest = {'ip': '172.22.112.251', 'port': 8086, 'db': 'healthresult', 'user': 'algtest', 'password': 'sensorweb711', 'ssl':False}
bsg_client = connect_influxdb(bsg_dest)

def upload_for_one_patient(influxdb_client, file_path = './Orth/features/'):
    #read_result
    data = np.load(file_path)
    sig_ = data[:,:1280]
    labels = data[:, 1280:]
    mac_address = int_to_mac(labels[0, 1])
    patient_ID = labels[0, 0]
    time_list = labels[:, 2]
   

    #merge heartbeat of one timestamp to one label
    cnt = 0
    json_body = []
    pred_hr = []
    gt_hr = []
    bsg_hr = []
    print(patient_ID, mac_address, switch_UNX_to_NY_Time(min(time_list)), switch_UNX_to_NY_Time(max(time_list)))
    for idx in range(len(time_list)):
        cur_time = time_list[idx]
        cur_timestamp = switch_UNX_to_NY_Time(cur_time)
        cur_x = sig_[idx]
        cur_Art_hr = compute_hr(cur_x)
        if cur_Art_hr is None:
            continue

        cur_label_hr = labels[idx, 3:13]
        cur_gt_hr = [x for x in cur_label_hr if x>0]
        if len(cur_gt_hr) ==0:
            continue
        else:
            cur_gt_hr = np.median(cur_gt_hr)
        gt_hr.append(cur_gt_hr)
        field_dict = {
            'ABP_HR': float(cur_Art_hr),
        }
        pred_hr.append(cur_Art_hr)
        # print(cur_timestamp, cur_Art_hr)
        start_timestamp = cur_timestamp - timedelta(seconds=0.5)
        end_timestamp = cur_timestamp + timedelta(seconds=0.5)
        cur_bsg_hr = read_influx(bsg_client, mac_address, 'vitals', 'heartrate', int(start_timestamp.timestamp()), int(end_timestamp.timestamp()))
        if len(cur_bsg_hr)==0:
            bsg_hr.append(-1)
        else:
            bsg_hr.append(cur_bsg_hr[0])
            field_dict['BSG_HR'] = float(cur_bsg_hr[0])

        #Upload part
        cnt += 1
        #start_timestamp: datetime format
        json_point = {
            "measurement": 'Test',
            "tags": {'location': mac_address},  # Extract tags
            "time": cur_timestamp.isoformat(timespec='microseconds'),
            "fields": field_dict
            # Extract fields
        }
        json_body.append(json_point)

        if (cnt % MAX_POINT_PER_WRITE) == 0:
            data_msg = json_body.copy()
            msg_queue.put(data_msg)
            json_body.clear()
            sleep(0.1)  # Adjust this value based on the CPU load on the server. Make sure the CPU load is below 70%
            print(f'Add {MAX_POINT_PER_WRITE} / {cnt}')
           

    if json_body:
        msg_queue.put(json_body)
        print(f'Add {len(json_body)} to the end')
    pred_hr = np.array(pred_hr)
    gt_hr = np.array(gt_hr)
    bsg_hr = np.array(bsg_hr)
    return pred_hr, gt_hr, bsg_hr
   

SUBS = [130, 101, 23, 98, 91, 144, 65, 24, 42, 134, 28, \
              131, 15, 12, 62, 129, 113, 13, 66, 81, 82, 151, \
              116, 136, 37, 59, 105, 67, 26, 43, 95, 111, 16, \
              68, 85, 96, 45, 120, 132, 114, 106, 93, 123,\
              78, 150, 29, 38, 109, 46, 64, 118, 70, 35]
SUBS.sort()
if __name__ == '__main__':

    dest = {'ip': '172.22.112.251', 'port': 8086, 'db': 'icu', 'user': 'algtest', 'password': 'sensorweb711', 'ssl':False}
    influxdb_client = connect_influxdb(dest)
    write_db_thread = threading.Thread(target=write_db, args=(influxdb_client,))
    write_db_thread.daemon = True
    write_db_thread.start()

    Art_path = '/media/test/Expansion/Patients_Collection/'
    res_dict = {}
    for i in SUBS:
        file_path = os.path.join(Art_path, f'ID_{i}', 'NPY', '10s', 'Matched_NPY', f'ID_{i}_10s_Art_Labels.npy')
        if os.path.isfile(file_path):
            print(f"Processing Patient {i}...")
            pred_hr, gt_hr, bsg_hr = upload_for_one_patient(influxdb_client=influxdb_client, file_path = file_path)
            res_dict[int(i)] = [np.mean(np.abs(pred_hr-gt_hr))]
            print(pred_hr.shape, gt_hr.shape,bsg_hr.shape)
            mask = bsg_hr > 0
            if sum(mask)==0:
                res_dict[int(i)].append(np.nan)
            else:
                bsg_hr = bsg_hr[mask]
                gt_hr = gt_hr[mask]
                res_dict[int(i)].append(np.mean(np.abs(bsg_hr-gt_hr)))
           
            print(f"Patient {i} Finished.\n")
        else:
            print(f"Patient {i} does not have Art data.")

           

    influxdb_client.close()
    print('over')
 
    for k, v in res_dict.items():
        print(k, v)