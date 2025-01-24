import os, sys, yaml, argparse, uuid, queue, threading, time, struct, enum, logging, json
from multiprocessing import Queue
import paho.mqtt.client as mqtt
from time import sleep
import numpy as np

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

def test_match_shift(bsg_long, ecg_long):
    return


def signal_check(mac_addr, seismic_data_queue):

    buffersize = 600 # second
    samplingrate_ecg = 128
    samplingrate_bsg = 100
    BUFFER_SIZE_MAX_ecg = int(buffersize) * int(samplingrate_ecg)
    BUFFER_SIZE_MAX_bsg = int(buffersize) * int(samplingrate_bsg)
    raw_data_buf_ecg=[]
    raw_data_buf_bsg=[]

    while True:
        try:
            msg=seismic_data_queue.get(timeout=300) # timeout 5 minutes
            logger.debug(f'The queue of mac {mac_addr} id {id(seismic_data_queue)} lose a seismic_data. The length of it is {seismic_data_queue.qsize()}')
        except queue.Empty: #if timeout and does't receive a message, remove mapping dictionary and exit current thread
            print(f"{mac_addr} have not received message for 5 minute, process terminated")
            break

        if (msg is None) or ("timestamp" not in msg) or ("data_interval" not in msg) or ("data" not in msg):  # If None is received, break the loop
            print(f"Process {mac_addr}  Received wrong seismic data. exit")
            break
        timestamp=msg["timestamp"]
        data_interval=msg["data_interval"]
        data=msg["data"]
        signal_type = msg["signal_type"]
        # print('mac:', mac_addr, 'timestamp:', timestamp, 'data len:', len(data))
        
        if len(data) < 100:
            data = np.pad(data, (0, 100-len(data)), 'constant', constant_values=(np.mean(data), np.mean(data)))

        if signal_type == 'ecg':
            raw_data_buf_bsg += data
        elif signal_type == 'bsg':
            raw_data_buf_ecg += data
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        buf_len_bsg=len(raw_data_buf_bsg)
        buf_len_ecg=len(raw_data_buf_ecg)
        # logger.debug(f'The length of raw_data_buf is {buf_len}')
        # logger.debug(f'The length of raw_data_buf is {buf_len}, the last value of data is {data[-1]}')

        if((buf_len_bsg > BUFFER_SIZE_MAX_bsg - 100) and (buf_len_ecg > BUFFER_SIZE_MAX_ecg - 128)):
            np.save(f'../data/ecg_{timestamp}', np.array(raw_data_buf_ecg))
            np.save(f'../data/bsg_{timestamp}', np.array(raw_data_buf_bsg))

            bsg_long = np.array(raw_data_buf_bsg).copy()
            ecg_long = np.array(raw_data_buf_ecg).copy()
            del raw_data_buf_bsg[0:buf_len_bsg]
            del raw_data_buf_ecg[0:buf_len_ecg]

            #prep work for downstream tasks
            try:
                match_or_not, time_shift = test_match_shift(bsg_long, ecg_long)
            except Exception as e:
                logger(f"MAC={mac_addr}: AI predict function ERROR,Terminated: {e}")
                break
            
            result={
                "mac_addr": mac_addr,
                "match_or_not": match_or_not,
                "time_shift": time_shift,
                "timestamp": timestamp,
                # "alert":alert,
            }
            try:
                result_queue.put(result)
            except Exception as e:
                logger.info(f"MAC={mac_addr}: Send vital ERROR,Terminated: {e}")
                break
    return

def send_vital_result(group, mac, timestamp, match_or_not, time_shift, alert):
    topic="/" + group + "/" + mac + "/match_or_not"
 
    payload = "timestamp=" + str(timestamp) + "; match_or_not=" + str(match_or_not) + "; time_shift=" + str(time_shift) + "; alert=" + str(alert)

    mqtt_dedicated_publisher.publish(topic, payload, qos=1)
    # if debug:
    #     global mqtt_publishing_cnt
    #     mqtt_publishing_cnt +=1
    #     if mqtt_publishing_cnt % 100 ==0:
    #         print(f"mqtt_publishing_cnt={mqtt_publishing_cnt}")
    # else:
    #     # print(topic)
    #     pass
    return

def publish_result(mqtt_msg_q):

    while True:
        #get result data message 
        try:
            msg=mqtt_msg_q.get() 
        except Exception as e:
            logger.info(f"process_schedule(), failed to get mqtt message from queue. error={e}")
            sys.exit()
        if None == msg:
            continue

        mac_addr=msg.get("mac_addr")
        if (None == mac_addr):
            logger.info(f"Missing mac_addr in the message")
            continue

        timestamp=msg.get("timestamp",0)   
        match_or_not=msg.get("match_or_not", -20)
        time_shift=msg.get("time_shift", 0)
        alert=msg.get("alert", -1)   
        # timestamp=(int(timestamp * 10**9) // 10**7) * 10**7

        group=mq_map.get_group(mac_addr)
        if group:
            send_vital_result(group, mac_addr ,timestamp, match_or_not, time_shift, alert)


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
                if proc.is_alive() and backlog < 180:
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


def setup_mqtt_for_publishing_data(config_dict):
    return

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
    mqtt_dedicated_publisher, mqtt_thread_pub=setup_mqtt_for_publishing_data(config_dict)
    thread_list.append(mqtt_thread_recv)
    logger.info(
        f"The receiver thread is added to thread list\n"
        f"The thread_list has:\n" +
        "\n".join([thread.name for thread in thread_list])
    )
    # Create a dedicated MQTT client for publishing vital data
    # mqtt_dedicated_pubish, mqtt_thread_pub=setup_dedicated_mqtt_for_pubishing_data(mqtt_conf)
    # thread_list.append(mqtt_thread_pub)

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
    # publish_thread = threading.Thread(target=publish_result, args=(result_queue,) )
    # publish_thread.daemon = True
    # publish_thread.start()
    # thread_list.append(publish_thread)

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