'''
NOTE:
This file is the interface between the framework and the AI module. Which involvs two functions:

1. setup_args_for_ai()
    This function is used to set arguments in the command line. AI developers can add command line arguments to this function as needed.
    All the argruments will be passed to ai_unit_process via "**kwargs". 
    After using "args = argparse.Namespace(**kwargs)" to convert them, developers can access arguments just as they used to.

    This function will be called once during framework startup.

2. ai_unit_process(mac_addr, seismic_data_queue, vital_queue, **kwargs):
    This function will be run as a independent process for a single device. 

    mac_addr, MAC address for a Beddot device.

    seismic_data_queue, is a queue used to recieve seismic data, structured in a dictionary format as shown below. All data has been extracted from the MQTT message.
        seismic_data={
        “timestamp”:			# in nano seconds
        “data_interval”:		# in nano seconds
        “data”:		# a list with data points
        }

    vital_queue, is a queue used to return results from the AI engine to the framework. Messages for the result are structured in a dictionary format as below:
        result={
            "mac_addr": mac_addr,
            "hr":hr,
            "rr":rr,
            "bph":bph,
            "bpl":bpl,
            "mv":mv,
            "vital_timestamp":vital_timestamp,              # in seconds
            "oc":oc,
            "occupancy_timestamp":occupancy_timestamp,      #in seconds
            "alert":alert,                                  # is a number
            "alert_timestamp":alert_timestamp               #in seconds
        }

    **kwargs, settings that are from command line, database, CSV file and Yaml file are passed via this argument.
        --kwargs["command_line_args"], command_line_args is the key word set by parser.add_argument() in the setup_args_for_ai() function
        --kwargs["alert_settings"], the alert setting for "mac_addr". 
        --kwargs["version"], the version setting for "mac_addr". 
        --kwargs["csv_conf"], the original parameter from CSV file. Developers can add fields to the CSV file as needed, which will be passed via this argument.
             The "alert_setting" and "monitoring_target" fields in CSV file are parsed and passed by kwargs["alert_settings"],kwargs["version"]. 
             So, if you don't have additional settings in CSV file, you don't need to access kwargs["csv_conf"]. 

             kwargs["csv_conf"] is a dictionary with MAC address as keyword. e.g.
             ai_kwargs[csv_conf]={'device_id': '305', 'device_mac': '74:4d:bd:89:2d:5c', 'ai_active': '1', 'monitoring_target': 'adult', 'alert_setting': '{ "hr": {"max": 120, "min": 45}, "rr": {"max": 22, "min": 10}, "bph": {"max": 140, "min": 90}, "bpl": {"max": 90, "min": 60}, "oc": {"on": 1, "off": 1} }'}
             

        Use "args = argparse.Namespace(**kwargs)" to convert it to namespace, then use "args.keyword" to access, e.g. args.version

        

Additionally,
1) when setting the device list in Yaml file, '*' can be used as a wildcard, matching all devices, see yaml file. No need to add any code to the framework
2) Define the path to CSV file in Yaml file. You can add or delete columns except "device_mac","monitoring_target","alert_setting"
'''


import numpy as np
# from framework_adapter import *

from collections import deque
import queue
import math
import argparse
from common import logger


def dummy_load():
    j=0
    for i in range(100000):
        j=j+1
        v=i*j
        k=v/45.98
    return(k)

#======================== setting up command line ====================================

def setup_args_for_ai():
    '''
    This function will be called by the framework during startup. 
    All command-line parameters will pass to the ai_unit_process() via kwargs
    '''
    parser = argparse.ArgumentParser(description='BedDot - Sleep Activities and Vital Signs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The '-c' argument must be include.
    parser.add_argument('-c','--conf_file', type=str, default='ai_mqtt_conf.yaml',
                    help='the ai yaml conf file', required=True)
    parser.add_argument('-t','--thread', action=argparse.BooleanOptionalAction)
    # parser.add_argument('-t','--task', type=str, default='process',
    #                 help='thread for multi-thread or process for multi-process')
    
    #parser.add_argument("dot", type=str, help='dot2.us')
    parser.add_argument('--vitals', type=str, default='HRSD', help='the vitals to calculate')
    parser.add_argument('--algo_name', type=str, default='algo_DSPYS', 
                        help='the default algorithm name')
    parser.add_argument('--algo_bp', type=str, default='algo_VTCN', #'algo_LSTMAttention', #
                        help='the default BP model name')
    # parser.add_argument('--debug', type=str, default='False',
    #                     help='the debug mode: True/False')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--allow_all', action=argparse.BooleanOptionalAction)
    parser.add_argument('--version', type=str, default='adult',
                        help='the algorithm version: adult/animal/baby')
    # parser.add_argument('--rr_duration', type=int, default=45, help='rr duration')
    parser.add_argument('--list_file', type=str, default='', help='the live run list file')
    parser.add_argument('--oc_v', type=str, default='adult_dl', help='the occupancy version: adult_dl/adult_dsp/animal_dsp')

    args = parser.parse_args()
    return args


#======================== Seperated Precess Zone ====================================
# Important Reminder:
# All the functions below are accessed in a seperated process. 
# Therefore, global variable is not available due to the memory isolation with multiprecessing

def predict(args, buffer, timestamp, mac_addr, alert_settings, ai_data_buf):
   
    # print(f"mac={mac_addr}, alert_settings={alert_settings}")
    # print(f"You can also get alert_settings like this: args.alert_settings={args.alert_settings}")
    # print(f"You can get version like this: args.version={args.version}")

    # add algorithm details
    # dummy_load()
    
    hr=np.random.randint(60, 90, 1)
    rr=np.random.randint(10, 20, 1)
    bph=np.random.randint(120, 150, 1)
    bpl=np.random.randint(90, 120, 1)
    oc=[1,-1,-1,-1]
    mv=1
    vital_timestamp=timestamp
    occupancy_timestamp=timestamp
    alert=1
    alert_timestamp=timestamp

    return hr[0],rr[0],bph[0],bpl[0],mv,vital_timestamp, oc, occupancy_timestamp, alert, alert_timestamp



def ai_unit_process(mac_addr, seismic_data_queue, vital_queue, **kwargs):
    
    args = argparse.Namespace(**kwargs)

    # print(f"mac={mac_addr}, alert={args.alert_settings}, version={args.version}")
    
    buffersize   = 60 # config.get('general', 'buffersize')
    samplingrate = 100 # int(config.get('general', 'samplingrate'))
    hrTimeWindow    = 30 # int(config.get('main', 'hrTimeWindow'))
    BUFFER_SIZE_MAX = int(buffersize) * int(samplingrate)
    WINDOW_SIZE = elementsNumberHR = hrTimeWindow * samplingrate
    raw_data_buf=[]
    ai_data_buf={}

    while True:
        #get raw data message from queue
        try: 
            msg=seismic_data_queue.get(timeout=300) # timeout 5 minutes
        except queue.Empty: #if timeout and does't receive a message, remove mapping dictionary and exit current thread
            logger(f"{mac_addr} have not received message for 5 minute, process terminated")
            break

        if (msg is None) or ("timestamp" not in msg) or ("data_interval" not in msg) or ("data" not in msg):  # If None is received, break the loop
            logger(f"Process {mac_addr}  Received wrong seismic data. exit")
            break
        timestamp=msg["timestamp"]
        data_interval=msg["data_interval"]
        data=msg["data"]
 
        raw_data_buf += data
        buf_len=len(raw_data_buf)
        # if debug: print(WINDOW_SIZE, BUFFER_SIZE_MAX, buf_len)
        # dump overflow data
        if(buf_len > BUFFER_SIZE_MAX):
            difSize = buf_len - BUFFER_SIZE_MAX
            del raw_data_buf[0:difSize]

        if buf_len < WINDOW_SIZE :
            continue

        #prep work for AI, and call Ai algrithm
        data = raw_data_buf
        alert_settings=kwargs.get("alert_settings")
        try:
            hr,rr,bph,bpl,mv,vital_timestamp, oc,occupancy_timestamp, alert, alert_timestamp = predict(args, data, math.floor(timestamp/10**9), mac_addr, alert_settings, ai_data_buf)
        except Exception as e:
            logger(f"MAC={mac_addr}: AI predict function ERROR,Terminated: {e}")
            break
        
        result={
            "mac_addr": mac_addr,
            "hr":hr,
            "rr":rr,
            "bph":bph,
            "bpl":bpl,
            "mv":mv,
            "vital_timestamp":vital_timestamp,
            "oc":oc,
            "occupancy_timestamp":occupancy_timestamp,
            "alert":alert,
            "alert_timestamp":alert_timestamp
        }
        try:
            vital_queue.put(result)
        except Exception as e:
            logger(f"MAC={mac_addr}: Send vital ERROR,Terminated: {e}")
            break
    return