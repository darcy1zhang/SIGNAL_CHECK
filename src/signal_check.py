import os, sys, yaml, argparse, uuid, queue, threading, time, struct, enum, logging, json
from multiprocessing import Queue
import paho.mqtt.client as mqtt
from time import sleep

logging.basicConfig(
    filename="test.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MULTI_THREAD = 1

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

def signal_check(mac_addr, seismic_data_queue):

    buffersize = 600 # second
    samplingrate = 100
    BUFFER_SIZE_MAX = int(buffersize) * int(samplingrate)
    raw_data_buf=[]

    while True:
        try:
            msg=seismic_data_queue.get(timeout=300) # timeout 5 minutes
        except queue.Empty: #if timeout and does't receive a message, remove mapping dictionary and exit current thread
            print(f"{mac_addr} have not received message for 5 minute, process terminated")
            break

        if (msg is None) or ("timestamp" not in msg) or ("data_interval" not in msg) or ("data" not in msg):  # If None is received, break the loop
            print(f"Process {mac_addr}  Received wrong seismic data. exit")
            break
        timestamp=msg["timestamp"]
        data_interval=msg["data_interval"]
        data=msg["data"]
        print('mac:', mac_addr, 'timestamp:', timestamp, 'data len:', len(data))

        raw_data_buf += data
        buf_len=len(raw_data_buf)

        if(buf_len > BUFFER_SIZE_MAX - 100):
            difSize = buf_len - BUFFER_SIZE_MAX
            del raw_data_buf[0:difSize]
        
        data = raw_data_buf
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

def process_schedule(mqtt_msg_q):
    latest_chk_time=time.monotonic()
    while True:
        #get raw data message 
        msg=mqtt_msg_q.get() 

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
        mac_addr,timestamp, data_interval, signal_type, data = parse_beddot_data(msg)
        seismic_data={"timestamp":timestamp, "data_interval":data_interval, "data":data}

        mq_id=mq_map.get_ai_input_q(mac_addr)
        if (None == mq_id): 
            # print(f"dot_license.number_of_devices()={dot_license.number_of_devices()}")
            if mq_map.get_number_of_queue() < 50:
                ai_input_q = Queue()

                # Launch a new process/thread
                if MULTI_THREAD:
                    p = threading.Thread(target=signal_check, args=(mac_addr, ai_input_q,), name=mac_addr)
                    p.daemon = True

                p.start()

                # Add info to the mapping centre for further use.
                mq_map.add(mac_addr, p, ai_input_q, group)
            else:
                raise ValueError(f"Number of devices has reached the limit, please contact your provider")

        try:
            if mq_id:
                proc=mq_map.get_process_id(mac_addr)
                backlog=mq_id.qsize()
                if proc.is_alive() and backlog < 180:
                    mq_id.put(seismic_data)
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
                client.subscribe(t,qos=1)

def on_mqtt_message(client, userdata, msg):
    mqtt_msg_queue.put(msg)
    return

def setup_mqtt_for_raw_data(config_dict):

    # Create MQTT client for receiving raw data
    unique_client_id = f"AI_Subscribe_{uuid.uuid4()}"
    mqtt_client = mqtt.Client(client_id=unique_client_id)
# , callback_api_version = CallbackAPIVersion.VERSION2

    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect(config_dict["mqtt"]['ip'], config_dict['mqtt']["port"], 60)
    mqtt_thread = threading.Thread(target=lambda: mqtt_client.loop_forever()) 
    mqtt_thread.daemon = True
    mqtt_thread.start()
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
    logging.info(json.dumps(data, ensure_ascii=False))
    mqtt_dedicated_receive, mqtt_thread_recv=setup_mqtt_for_raw_data(config_dict)
    thread_list.append(mqtt_thread_recv)
    # Create a dedicated MQTT client for publishing vital data
    # mqtt_dedicated_pubish, mqtt_thread_pub=setup_dedicated_mqtt_for_pubishing_data(mqtt_conf)
    # thread_list.append(mqtt_thread_pub)

    # Luanch precess scheduling thread and result publishing thread
    schedule_thread = threading.Thread(target=process_schedule, args=(mqtt_msg_queue,) )
    schedule_thread.daemon = True
    schedule_thread.start()
    thread_list.append(schedule_thread)

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