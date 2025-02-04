import os, sys, yaml, argparse, uuid, queue, threading, time, struct, enum, logging, json
from multiprocessing import Queue
import paho.mqtt.client as mqtt
from time import sleep
import numpy as np
import bisect
from BCG_preprocessing import signal_quality_assessment_v4
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor
from influxdb import InfluxDBClient
import concurrent.futures


def get_hr_bsg(bsg_10s):
    hr_bsg = []
    fs_bsg = (bsg_10s.shape[1] - 1) / 10
    for i in range(bsg_10s.shape[0]):
        sig = bsg_10s[i, 1:]
        res = signal_quality_assessment_v4(x = sig, Fs = fs_bsg,
                                            n_lag = len(sig)//2,
                                            low = 1, high = 20,
                                            acf_low = int(0.25 * fs_bsg),
                                            acf_high = int(1.2 * fs_bsg),
                                            f_low = 0.6)
        if not res[0]:
            res = signal_quality_assessment_v4(x = sig, Fs = fs_bsg,
                                            n_lag = len(sig)//2,
                                            low = 1, high = 10,
                                            acf_low = int(0.25 * fs_bsg),
                                            acf_high = int(1.2 * fs_bsg),
                                            f_low = 0.6, diff = 0)
        if res[0]:
            hr_bsg.append(np.array([bsg_10s[i, 0], res[-1]]))

    hr_bsg = np.array(hr_bsg)
    return hr_bsg

def get_hr_ecg(ecg_10s):
    hr_ecg = []
    for i in range(ecg_10s.shape[0]):
        hr_ecg.append(np.array([ecg_10s[i, 0], 80 + np.random.randint(-5, 5)]))
    hr_ecg = np.array(hr_ecg)
    return hr_ecg

def unix_2_et(timestamp):

    utc_time = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)
    eastern_time = utc_time.astimezone(pytz.timezone('US/Eastern'))
    date_str = eastern_time.strftime('%Y_%m_%d_%H_%M_%S')
    return date_str

def get_10s_data(signal, signal_type):
    if signal is None:
        return None

    # signal: (n * samplingrate + 1)
    if signal_type == 'ecg':
        fs = 256
    elif signal_type == 'bsg':
        fs = 100
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    start_time = round(signal[0][0] // 10 * 10)
    end_time = round(signal[-1][0] // 10 * 10)
    signal_10s = []

    for i in range(start_time, end_time + 1, 10):
        target_signal = signal[(signal[:, 0] >= i) & (signal[:, 0] < i + 10)]
        if target_signal.shape[0] == 10:
            signal_10s.append(np.concatenate([[round(target_signal[0, 0])], target_signal[:, 1:].flatten()]))
            # print(np.concatenate([[round(target_signal[0, 0])], target_signal[:, 1:].flatten()]).shape, 10 * fs + 1)
            assert np.concatenate([[round(target_signal[0, 0])], target_signal[:, 1:].flatten()]).shape == (10 * fs + 1, )

    if len(signal_10s) == 0:
        return None
    signal_10s = np.array(signal_10s)
    signal_10s_clean = []
    if signal_type == 'bsg':
        for i in range(signal_10s.shape[0]):
            sig = signal_10s[i, 1:]
            res = signal_quality_assessment_v4(x = sig, Fs = fs,
                                                n_lag = len(sig)//2,
                                                low = 1, high = 20,
                                                acf_low = int(0.25 * fs),
                                                acf_high = int(1.2 * fs),
                                                f_low = 0.6)
            if not res[0]:
                res = signal_quality_assessment_v4(x = sig, Fs = fs,
                                                n_lag = len(sig)//2,
                                                low = 1, high = 10,
                                                acf_low = int(0.25 * fs),
                                                acf_high = int(1.2 * fs),
                                                f_low = 0.6, diff = 0)
            if res[0]:
                signal_10s_clean.append(np.concatenate([[signal_10s[i, 0]], sig]))
        if len(signal_10s_clean) == 0:
            return None
        signal_10s_clean = np.array(signal_10s_clean)
        return signal_10s_clean

    return signal_10s
    

def find_max_index(arr, value):
    """Returns the max index where arr[index] <= value, or -1 if no such index exists."""
    index = bisect.bisect_right(arr, value) - 1
    return index if index >= 0 else -1

class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler("debug.log", mode="w")
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(DebugFilter())
info_handler = logging.FileHandler("info.log", mode="w")
info_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(threadName)s - %(message)s")
debug_handler.setFormatter(formatter)
info_handler.setFormatter(formatter)
logger.addHandler(debug_handler)
logger.addHandler(info_handler)
logger.debug('hello')

MULTI_THREAD = 1
result_queue=Queue()

class CallbackAPIVersion(enum.Enum):
    """Defined the arguments passed to all user-callback.

    See each callbacks for details: `on_connect`, `on_connect_fail`, `on_disconnect`, `on_message`, `on_publish`,
    `on_subscribe`, `on_unsubscribe`, `on_log`, `on_socket_open`, `on_socket_close`,
    `on_socket_register_write`, `on_socket_unregister_write`
    """
    VERSION1 = 1
    """The version used with paho-mqtt 1.x before introducing CallbackAPIVersion.

    This version had different arguments depending if MQTTv5 or MQTTv3 was used. `Properties` & `ReasonCode` were missing
    on some callback (apply only to MQTTv5).

    This version is deprecated and will be removed in version 3.0.
    """
    VERSION2 = 2
    """ This version fix some of the shortcoming of previous version.

    Callback have the same signature if using MQTTv5 or MQTTv3. `ReasonCode` are used in MQTTv3.
    """

def get_ecg_hr(ecg):
    return -20

def calc_match_score(hr_bsg, hr_ecg):
    diff = np.abs(hr_bsg - hr_ecg)
    if len(hr_bsg) == 0:
        return 1
    else:
        match_score = len(diff[diff < 10]) / len(diff)
    return match_score

def test_match_shift(hr_bsg, hr_ecg):
    if hr_bsg is None or hr_ecg is None:
        return 0, 0, [], []
    timestamps_bsg = hr_bsg[:, 0]
    timestamps_ecg = hr_ecg[:, 0]
    common_timestamps = np.intersect1d(timestamps_bsg, timestamps_ecg)

    # Step 3: Filter rows that match the common timestamps
    filtered_hr_bsg = hr_bsg[np.isin(hr_bsg[:, 0], common_timestamps)]
    filtered_hr_ecg = hr_ecg[np.isin(hr_ecg[:, 0], common_timestamps)]

    match_score = 0
    time_shift = 0
    match_score = calc_match_score(filtered_hr_bsg[:, 1], filtered_hr_ecg[:, 1])

    return match_score, time_shift, hr_bsg, hr_ecg


def signal_check(mac_addr, seismic_data_queue):

    buffersize = 500 # second
    samplingrate_ecg = 256
    samplingrate_bsg = 100
    # BUFFER_SIZE_MAX_ecg = int(buffersize) * int(samplingrate_ecg)
    # BUFFER_SIZE_MAX_bsg = int(buffersize) * int(samplingrate_bsg)
    raw_data_buf_bsg=[]
    raw_data_buf_ecg=[]
    
    executor = ThreadPoolExecutor(max_workers=4)
    while True:
        # print(len(raw_data_buf_bsg), len(raw_data_buf_ecg))
        try:
            msg=seismic_data_queue.get(timeout=300) # timeout 5 minutes
            logger.debug(f'The queue of mac {mac_addr} id {id(seismic_data_queue)} lose a seismic_data. The length of it is {seismic_data_queue.qsize()}')
        except queue.Empty: #if timeout and does't receive a message, remove mapping dictionary and exit current thread
            print(f"{mac_addr} have not received message for 5 minute, process terminated")
            break

        if (msg is None) or ("timestamp" not in msg) or ("data_interval" not in msg) or ("data" not in msg):  # If None is received, break the loop
            print(f"Process {mac_addr}  Received wrong seismic data. exit")
            break
        timestamp=msg["timestamp"] // 1e9 # convert nano second to second
        data_interval=msg["data_interval"] # nano second
        data=msg["data"]
        signal_type = msg["signal_type"]
        # if signal_type == 'bsg':
        #     print(f"timestamp: {unix_2_et(timestamp)}")
        
        # if len(data) < 100:
        #     data = np.pad(data, (0, 100-len(data)), 'constant', constant_values=(np.mean(data), np.mean(data)))

        if signal_type == 'bsg':
            raw_data_buf_bsg.append(np.concatenate(([timestamp], data)))
        elif signal_type == 'ecg':
            raw_data_buf_ecg.append(np.concatenate(([timestamp], data)))
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        buf_len_bsg=len(raw_data_buf_bsg)
        buf_len_ecg=len(raw_data_buf_ecg)
        # print(f"bsg: {buf_len_bsg}, ecg: {buf_len_ecg}")
        # logger.debug(f'The length of raw_data_buf is {buf_len}')
        # logger.debug(f'The length of raw_data_buf is {buf_len}, the last value of data is {data[-1]}')


        if((buf_len_bsg > buffersize - 1) or (buf_len_ecg > buffersize - 1)):
            
            if buf_len_bsg == 0:
                raw_data_buf_bsg_npy = None
            else:
                raw_data_buf_bsg_npy = np.array(raw_data_buf_bsg)

            if buf_len_ecg == 0:
                raw_data_buf_ecg_npy = None
            else:
                raw_data_buf_ecg_npy = np.array(raw_data_buf_ecg)
            

            # start_timestamp = np.min([raw_data_buf_bsg[0][0], raw_data_buf_ecg[0][0]]) // 10 * 10
            # end_timestamp = np.max([raw_data_buf_bsg[-1][0], raw_data_buf_ecg[-1][0]]) // 10 * 10

            # bsg_tmp = raw_data_buf_bsg_npy[(raw_data_buf_bsg_npy[:, 0] >= start_timestamp) & (raw_data_buf_bsg_npy[:, 0] < end_timestamp)] # n * 1001
            # ecg_tmp = raw_data_buf_ecg_npy[(raw_data_buf_ecg_npy[:, 0] >= start_timestamp) & (raw_data_buf_ecg_npy[:, 0] < end_timestamp)]
            

            bsg_10s = executor.submit(get_10s_data, raw_data_buf_bsg_npy, 'bsg')
            ecg_10s = executor.submit(get_10s_data, raw_data_buf_ecg_npy, 'ecg')
            # print(future_bsg.result().shape)
            if bsg_10s.result() is not None:
                future_bsg_save = executor.submit(lambda: np.save(f'../data/bsg_{unix_2_et(bsg_10s.result()[0, 0])}', bsg_10s.result()))
                print(f'bsg shape: {bsg_10s.result().shape}')
            if ecg_10s.result() is not None:
                future_ecg_save = executor.submit(lambda: np.save(f'../data/ecg_{unix_2_et(ecg_10s.result()[0, 0])}', ecg_10s.result()))
                print(f'ecg shape: {ecg_10s.result().shape}')

            # bsg_tmp_10s = get_10s_data(raw_data_buf_bsg_npy, 'bsg')
            # ecg_tmp_10s = get_10s_data(raw_data_buf_ecg_npy, 'ecg')

            # if bsg_tmp_10s is not None:
            #     np.save(f'../data/bsg_{unix_2_et(bsg_tmp_10s[0, 0])}', bsg_tmp_10s)
            # if ecg_tmp_10s is not None:
            #     np.save(f'../data/ecg_{unix_2_et(ecg_tmp_10s[0, 0])}', ecg_tmp_10s)
                

            del raw_data_buf_bsg[0:len(raw_data_buf_bsg)]
            del raw_data_buf_ecg[0:len(raw_data_buf_ecg)]

            # prep work for downstream tasks
            try:
                hr_bsg = executor.submit(get_hr_bsg, bsg_10s.result())
                hr_ecg = executor.submit(get_hr_ecg, ecg_10s.result())
                match_thread = executor.submit(test_match_shift, hr_bsg.result(), hr_ecg.result())
                match_score = match_thread.result()
                # print(f'match_score: {match_score}, time_shift: {time_shift}')
                # match_score, time_shift, hr_bsg, hr_ecg = test_match_shift(bsg_10s.result(), ecg_10s.result())
            except Exception as e:
                logger.info(f"MAC={mac_addr}: AI predict function ERROR,Terminated: {e}")
                break
            
            result={
                "mac_addr": mac_addr,
                "match_score": match_score,
                "timestamp": hr_bsg.result()[0, 0],
                "hr_bsg": hr_bsg.result(),
                "hr_ecg": hr_ecg.result(),
                # "alert":alert,
            }
            try:
                result_queue.put(result)
                # print('put result')
                logger.info(f"MAC={mac_addr}: Send vital data to queue, the length of it is {result_queue.qsize()}")
            except Exception as e:
                logger.info(f"MAC={mac_addr}: Send vital ERROR,Terminated: {e}")
                break
    return

def format_influxdb_data(tag_mac_addr, field_name, field_value, timestamp, signal_type, measurement):
    data_point = [
        {
            "measurement": measurement,
            "tags": {"location": tag_mac_addr + "_" + signal_type},
            "time": int(timestamp * 1e9),
            "fields": {field_name: field_value}
        }
    ]
    return data_point

def pack_hr_data_for_influx(mac_addr, data, db_measurement, signal_type, db_field='value'):
    data_pionts=[]
    for i in range(data.shape[0]):
        point = format_influxdb_data(mac_addr, db_field, float(data[i, 1]), data[i, 0], signal_type, db_measurement)
        data_pionts +=point
    return data_pionts

def pack_match_score_data_for_influx(mac_addr, timestamp, data, db_measurement, signal_type, db_field='value'):
    point = format_influxdb_data(mac_addr, db_field, data, timestamp, signal_type, db_measurement)
    return point

def unpack_msg(msg):
    mac_addr=msg.get("mac_addr")
    match_score=msg.get("match_score")
    # time_shift=msg.get("time_shift")
    timestamp=msg.get("timestamp")
    hr_bsg=msg.get("hr_bsg", None)
    hr_ecg=msg.get("hr_ecg", None)

    db_measurement = "signal_check"
    data_points_hr_bsg = pack_hr_data_for_influx(mac_addr, hr_bsg, db_measurement, signal_type = 'hr_bsg')
    data_points_hr_ecg = pack_hr_data_for_influx(mac_addr, hr_ecg, db_measurement, signal_type = 'hr_ecg')
    data_points_match_score = pack_match_score_data_for_influx(mac_addr, timestamp, match_score, db_measurement, signal_type = 'match_score')
    data_pionts = data_points_hr_bsg + data_points_hr_ecg + data_points_match_score
        
    return data_pionts


executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def execute_asyn_done_cb(fut):
    try:
        result = fut.result(timeout=10)
        # if debug: print("Write dateabase successful (callback):", result)
    except Exception as e:
        logger.info(f'Exception:{e}, execute_asyn_done_cb')
def execute_asyn(func,param):
    try:
        # summit function to the executor
        future = executor.submit(func, param)
        future.add_done_callback(execute_asyn_done_cb)

    except concurrent.futures.ThreadPoolExecutor as e:
        logger.info(f"ThreadPoolExecutor error: {e}")

def write_influxdb(result_queue):
    
    url_ip = "localhost"
    influxdb_client = InfluxDBClient(host=url_ip, port=8086, database="shake")

    # Check the connection status
    try:
        if influxdb_client.ping():
            logger.info(f"Successfully connected to the Influxdb! {url_ip}")
            # print("")
        else:
            logger.info("Failed to connect to the Influxdb, exit")
            exit(1)
    except Exception as e:  
        logger.info("Ping failure, Connection to the Influxdb failed !, exit")
        exit(1)

    # Creating a thread pool for executing writes to influxDB
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # Using a buffer to collect data points, in order to reduce write operaton to InfluxDB
    influxdb_data_pionts_buf = []
    last_save_time = time.monotonic()
        
    while True:
        msg_fetch=False
        try: 
            msg = result_queue.get(timeout=1) # timeout 1 second 
            # print(msg) 
            msg_fetch=True
        except queue.Empty:
            pass
        except Exception as e:  
            logger.info(f"Failed to get item from the queue: {e}")
        
        if msg_fetch:
            data_pionts = unpack_msg(msg)
            # print(data_pionts)
            influxdb_data_pionts_buf += data_pionts

        now = time.monotonic()
        # write data when number of data point reaches a threshold or time up
        number_of_piont = len(influxdb_data_pionts_buf)
        # print(f"number_of_piont: {number_of_piont}")
        if (number_of_piont > 0) and (((now - last_save_time) > 1) or number_of_piont>=10) :
            last_save_time = now
            data_to_write = influxdb_data_pionts_buf.copy()
            # print(data_to_write[:2])
            influxdb_data_pionts_buf.clear()
            #add a writing task to thread pool
            execute_asyn(influxdb_client.write_points, data_to_write)
            # influxdb_client.write_points(data_to_write)
            # if debug: print("Data Point Writed:", number_of_piont)

        # check if "thread" is still avlive, if no quit the program
        if not is_all_threads_alive(thread_list): break
                 
    influxdb_client.close()
    sleep(2)
    logger.info("========= mqtt_fwd_influx exit ===========")
    sys.exit()

    return



class MessageQueueMapping:
    def __init__(self):
        self.msg_queue_dict = {}
        self.rw_lock = threading.RLock()
        
    def add(self, mac, process_id, ai_input_q, group):
        with self.rw_lock:
            if self.msg_queue_dict.get(mac) is not None:
                print("Duplicate entry. Key '{}' already exists.".format(mac))
            else:
                self.msg_queue_dict[mac] = [process_id, ai_input_q, group]
                
    def remove(self, mac):
        with self.rw_lock:
            try:
                if mac in self.msg_queue_dict:
                    self.msg_queue_dict.pop(mac)
            except Exception as e:
                pass
                
    def get_ai_input_q(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[1] if id is not None else None
    
    # def get_ai_output_q(self, mac):
    #     with self.rw_lock:
    #         id=self.msg_queue_dict.get(mac)
    #     return id[2] if id is not None else None
    
    def get_group(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[2] if id is not None else None
    
    def get_process_id(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[0] if id is not None else None
    
    def get_number_of_queue(self):
        return len(self.msg_queue_dict)

    def terminate_all_process(self):
        with self.rw_lock:
            for mac, id in self.msg_queue_dict.items():
                proc=id[0]
                q=id[1]
                if proc.is_alive():
                    while not q.empty():
                        try:
                            q.get_nowait()  # Non-blocking
                        except queue.Empty:
                            break
                    q.put(None)     # Send a None message to notify the thread to terminate 
                    if MULTI_THREAD:
                        proc.join()
                    else:
                        proc.terminate()

    def update_proc_status(self):
        temp={}
        with self.rw_lock:
            temp=self.msg_queue_dict.copy()

        for mac, id in temp.items():
            proc_id=id[0]
            if not proc_id.is_alive():
                with self.rw_lock:
                    self.msg_queue_dict.pop(mac)

mqtt_msg_queue=queue.Queue()
mq_map = MessageQueueMapping()

def parse_beddot_data(msg):
    bytedata=msg.payload
    mac_addr = ":".join(f"{x:02x}" for x in struct.unpack("!BBBBBB", bytedata[0:6]))
    data_len =struct.unpack("H",bytedata[6:8])[0]
    timestamp=struct.unpack("L",bytedata[8:16])[0]  # in micro second
    data_interval=struct.unpack("I",bytedata[16:20])[0]  # in micro second
    # sampling_rate = struct.unpack("I",bytedata[16:20])[0]  # in micro second

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
    return  mac_addr, timestamp, data_interval, data

def process_schedule(mqtt_msg_q):
    latest_chk_time=time.monotonic()
    while True:
        #get raw data message 
        msg=mqtt_msg_q.get() 
        logger.info(f'Get a message out of the queue, the length of it is {mqtt_msg_q.qsize()}')

        now=time.monotonic() 
        if now -  latest_chk_time > 30:
            mq_map.update_proc_status()
            latest_chk_time=now  
            
        if None == msg:
            continue

        # extract group name from topic
        group="Unnamed"
        substrings = msg.topic.split("/")
        # print(f"substrings={substrings}")
        for substr in substrings[1:]:
            if substr:
                group=substr
                break

        # extact information from the message and put them into a buffer
        mac_addr, timestamp, data_interval, data = parse_beddot_data(msg)
        seismic_data={"timestamp":timestamp, "data_interval":data_interval, "signal_type":substrings[3], "data":data }

        mq_id=mq_map.get_ai_input_q(mac_addr)
        if (None == mq_id): 
            # print(f"dot_license.number_of_devices()={dot_license.number_of_devices()}")
            if mq_map.get_number_of_queue() < 50:
                ai_input_q = Queue()
                logger.info(f'The mac {mac_addr} has initialized a queue to store the input. The len of the queue {id(ai_input_q)} is {ai_input_q.qsize()}')

                # Launch a new process/thread
                if MULTI_THREAD:
                    p = threading.Thread(target=signal_check, args=(mac_addr, ai_input_q,), name=mac_addr)
                    p.daemon = True

                p.start()
                logger.info(f'The thread to process mac {mac_addr} is created. The thread id is {p.ident} name is {p.name}')
                # Add info to the mapping centre for further use.
                mq_map.add(mac_addr, p, ai_input_q, group)
                proc=mq_map.get_process_id(mac_addr)
                mq_id = mq_map.get_ai_input_q(mac_addr)
                mq_id.put(seismic_data)
                logger.debug(f'The queue of mac {mac_addr} id {mq_id} is added a new seismic_data. The length of it is {mq_id.qsize()}')
            else:
                raise ValueError(f"Number of devices has reached the limit, please contact your provider")

        try:
            if mq_id:
                proc=mq_map.get_process_id(mac_addr)
                backlog=mq_id.qsize()
                if proc.is_alive() and backlog < 1800:
                    mq_id.put(seismic_data)
                    # print(f'fffffffff {seismic_data["data"][-1]}')
                    logger.debug(f'The queue of mac {mac_addr} id {mq_id} is added a new seismic_data. The length of it is {mq_id.qsize()}')
                else:
                    if proc.is_alive():
                        if MULTI_THREAD:
                            # Clear the queue
                            while not mq_id.empty():
                                try:
                                    mq_id.get_nowait()  # Non-blocking
                                except queue.Empty:
                                    break
                            mq_id.put(None)     # Send a None message to notify the thread to terminate 
                        else:
                            proc.terminate()

                        raise ValueError(f"Backlog on MAC={mac_addr},{backlog} messages. Process Terminated")
                    mq_map.remove(mac_addr)
                    # logger(f"Task for MAC={mac_addr} Terminated")
        except Exception as e:
            raise ValueError(f"Failed to put a raw data message into the queue for mac: {mac_addr}. error={e}")

def parse_config_file(yaml_file):

    config_yaml_file=os.path.abspath(os.path.expanduser(yaml_file))

    # Check if the configration file exists
    if not os.path.exists(config_yaml_file):
        sys.exit(f"Error: Config file {config_yaml_file} not found.")

    config_dict = {}
    with open(config_yaml_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict

def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    topics_str = config_dict['mqtt']['topic_filter']
    if topics_str is not None:
        # Remove the { } pairs
        topics_str=topics_str.replace("{", "")
        topics_str=topics_str.replace("}", "")
        # Delete the leading and trailing whitespace
        topics_str=topics_str.replace(" ", "")
        # topics_str=topics_str.strip()   
        topics=topics_str.split(",")
        # subscribe topic
        for t in topics:
            if t != "":
                print('subscribe topic:', t)
                client.subscribe(t)

def on_mqtt_message(client, userdata, msg):
    mqtt_msg_queue.put(msg)
    logger.info(f'Put a new message into the queue, the length of it is {mqtt_msg_queue.qsize()}')
    return

def setup_mqtt_for_raw_data(config_dict):

    # Create MQTT client for receiving raw data
    unique_client_id = f"AI_Subscribe_{uuid.uuid4()}"
    mqtt_client = mqtt.Client(client_id=unique_client_id)
    logger.info(f'The receiver client has been created. The client id is {unique_client_id}')
# , callback_api_version = CallbackAPIVersion.VERSION2

    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect(config_dict["mqtt"]['ip'], config_dict['mqtt']["port"], 60)
    logger.info(f'The receiver client has connected to ip {config_dict["mqtt"]["ip"]} port {config_dict["mqtt"]["port"]}')
    mqtt_thread = threading.Thread(target=lambda: mqtt_client.loop_forever(), name='mqtt_clinet_receiver') 
    mqtt_thread.daemon = True
    mqtt_thread.start()
    logger.info(f'The receiver thread is bind to thread id {mqtt_thread.ident} name {mqtt_thread.name}')
    return mqtt_client, mqtt_thread

def is_all_threads_alive(thd_list):
    alive=True
    for t in thd_list:
        if not t.is_alive():
            raise ValueError(f"thread {t} terminated")
            alive=False
            break
    return alive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signal Check Test', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The '-c' argument must be include.
    parser.add_argument('-c','--conf_file', type=str, default='ai_mqtt_conf.yaml',
                    help='the ai yaml conf file', required=True)
    
    args = parser.parse_args()
    thread_list=[]

    config_dict = parse_config_file(args.conf_file)
    logger.info(json.dumps(config_dict, ensure_ascii=False))
    mqtt_dedicated_receive, mqtt_thread_recv=setup_mqtt_for_raw_data(config_dict)
    # mqtt_dedicated_publisher, mqtt_thread_pub=setup_mqtt_for_publishing_data(config_dict)
    thread_list.append(mqtt_thread_recv)
    logger.info(
        f"The receiver thread is added to thread list\n"
        f"The thread_list has:\n" +
        "\n".join([thread.name for thread in thread_list])
    )

    # Luanch precess scheduling thread and result publishing thread
    logger.info(f'The length of message queue is {mqtt_msg_queue.qsize()}')
    schedule_thread = threading.Thread(target=process_schedule, args=(mqtt_msg_queue,), name='schedule_thread')
    schedule_thread.daemon = True
    schedule_thread.start()
    logger.info(f'The schedule thread is created id {schedule_thread.ident} name {schedule_thread.name}')
    thread_list.append(schedule_thread)
    logger.info(
        f"The schedule thread is added to thread list\n"
        f"The thread_list has:\n" +
        "\n".join([thread.name for thread in thread_list])
    )
    write_influxdb_thread = threading.Thread(target=write_influxdb, args=(result_queue,), name='write_influxdb_thread')
    write_influxdb_thread.daemon = True
    write_influxdb_thread.start()
    logger.info(f'The write_influxdb_thread thread is created id {write_influxdb_thread.ident} name {write_influxdb_thread.name}')
    thread_list.append(write_influxdb_thread)
    logger.info(
        f"The write_influxdb_thread thread is added to thread list\n"
        f"The thread_list has:\n" +
        "\n".join([thread.name for thread in thread_list])
    )

    loop_cnt=0
    while True:
        sleep(1)
        # check if "thread" is still avlive, if no quit the program
        if not is_all_threads_alive(thread_list): break
                
    mqtt_dedicated_receive.disconnect()  # Disconnect the MQTT client
    mqtt_dedicated_receive.loop_stop()   # Stop the loop
    # mqtt_dedicated_pubish.disconnect()
    # mqtt_dedicated_pubish.loop_stop()
    mqtt_thread_recv.join()        # Wait for the thread to finish
    # mqtt_thread_pub.join()        # Wait for the thread to finish

    mq_map.terminate_all_process()

    sys.exit()