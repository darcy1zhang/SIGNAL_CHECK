import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

import threading
# import struct
import queue
import yaml
from influxdb import InfluxDBClient
import time
from time import sleep
# import concurrent.futures
import sys,os
import colorama
from colorama import Fore, Back, Style
from common import *
from tls_psk_patch import set_tls_psk_patch
from msg_proc import *
from security import *
from authenticate import start_authenticate_thread, config_mem_cache
import uuid

# from databases.mysql_controller import mysql_db
current_dir=os.path.abspath(__file__)
project_package_dir = os.path.dirname(os.path.dirname(current_dir))
project_dir=os.path.dirname(project_package_dir)
project_dir_father=os.path.dirname(project_dir)

config_dir=os.getcwd()
# config_dir=os.path.join(project_dir_father, 'config')

sys.path.append(project_package_dir)
from interface.local_api import HomeDots

broker_psk_id="beddot"
broker_psk="70e14722dd92179cbb4f081bc68284eec83baad42b0654bf06ddd9d1ab29cd4c"

cache_conf_file_key=b'BMOx1fyro8oyP1plIlXlPIuk7Fldj2Hf5ca7oMEDPJM='
fwd_instance_name=''
fwd_instance_token=''

colorama.init(autoreset=True)

debug=False

msg_queue=queue.Queue()

def get_config_from_encrypt_file():

    cache_file_name = os.path.join(config_dir, f'.{fwd_instance_name}')
    config_info=decrypt_dict(cache_conf_file_key, cache_file_name)
    return config_info

def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    """
    Run when the client receives a CONNACK response from the server.
    :param client: the client instance for this callback
    :param userdata: the private user data as set in Client() or user_data_set()
    :param flags: response flags sent by the broker
    :param rc: the connection result
    :param properties: The properties returned from the broker
    :return:
    """
    if rc == 0:
        logger(f"{Fore.GREEN}Connected to MQTT broker {client._host} successfully: {rc}{Style.RESET_ALL}")
    else:
        logger(f"Connect failed with result code {rc}")
    
    # extract topics
    mqtt_conf=config_mem_cache.get_mqtt()

    topics_str=mqtt_conf.get("topic_filter")
    if topics_str is not None:
        # Remove the { } pairs
        topics_str=topics_str.replace("{", "")
        topics_str=topics_str.replace("}", "")
        # Delete the leading and trailing whitespace
        topics_str=topics_str.replace(" ", "")
        # topics_str=topics_str.strip()   
        topics=topics_str.split(",")
        # subscribe topic
        print(topics)
        for t in topics:
            if t != "":
                logger(f'subscribe topic: {t}')
                client.subscribe(t,qos=1)
    return

def on_mqtt_message(client, userdata, msg):
    # if (debug): print(f"Received message: {msg.topic}")
    mac=get_mac_from_topic(msg.topic)
    if mac == "" :
        mac=get_mac_from_payload_lagacy(msg.payload)
    
    if config_mem_cache.is_authentic_mac(mac):
        if (debug): print(f"Filtered message: {msg.topic}")
        try:
            msg_queue.put(msg)
        except Exception as e:
            logger(f"Failed to put raw data message into the queue: {e}")
    return
    
def check_cmd_line():
    global debug

    command_str=""
    config_yaml_file=""
    # check the number of argument 
    if len(sys.argv) >= 2:
        command_str = sys.argv[1]
        if len(sys.argv) >2:
            if "d"==sys.argv[2]:
                debug=True 
                print("----entering debug mode-----")
    else:
        print("")
        print("Usage: python3 mqtt_fwd_influx.py <'your_conf.yaml'>")
        print("Examples1: python3 mqtt_fwd_influx.py beddot_fwd_conf.yaml ")
        print("OR")
        print("Usage: mqtt_fwd_influx.bin <'your_conf.yaml'>")
        print("Examples1: mqtt_fwd_influx.bin beddot_fwd_conf.yaml ")
        exit(1)

    # Check if the configration file exists
    if "yaml" in command_str:
        if not os.path.exists(command_str):
            print("Error: Config file '{}' not found.".format(command_str))
            exit(1)
        
        config_yaml_file=os.path.abspath(os.path.expanduser(command_str))
    else :
        print(f"Configuratoin file must be ended with '.yaml'!")
        exit(1)

    # Create log file based on the command line input
    log_file_name=os.path.splitext(config_yaml_file)[0] + ".log"

    # load configuration from ai_com_conf.yaml
    config_info={}
    master_sever=""
    instance_name=""
    instance_scret=""
    config_file_info = get_config_info_from_file(config_yaml_file)
    if "config_mode" in config_file_info:
        if config_file_info["config_mode"] == "local" and "local" in config_file_info:
            config_info=config_file_info["local"]
        elif config_file_info["config_mode"] == "remote" and "remote" in config_file_info:
            if "master_server" in config_file_info["remote"] and \
                "instance_name" in config_file_info["remote"] and \
                "instance_scret" in config_file_info["remote"]:
                master_sever=config_file_info["remote"]["master_server"]
                instance_name=config_file_info["remote"]["instance_name"]
                instance_scret=config_file_info["remote"]["instance_scret"]
            else:
                print(f"Missing 'master_server' or 'instance_name' or 'instance_scret' in your yaml file")
                exit(1)
        else:
            print(f"'config_mode' error in your yaml file")
    else: 
        print(f"Missing 'config_mode' in your yaml file")
        exit(1)

    return config_info, log_file_name, master_sever, instance_name, instance_scret

def is_all_threads_alive(thd_list):
    alive=True
    for t in thd_list:
        if not t.is_alive():
            logger(f"thread {t} terminated")
            alive=False
            break
    return alive

if __name__ == '__main__':
    thread_list=[]
    # generate_key()
    config_info, log_file_name, master_server, fwd_instance_name, fwd_instance_token=check_cmd_line()

    config_dir=os.path.dirname(log_file_name)

    setup_logger(log_file_name)
    logger("========= mqtt_fwd_influx start ...===========")

    if not config_info:
        # launch a thread to upate authentic MAC addresses
        path_to_cache=os.path.dirname(log_file_name)
        authentication_thread=start_authenticate_thread(master_server, fwd_instance_name, fwd_instance_token, path_to_cache, "forwarder")
        thread_list.append(authentication_thread)

        # Awaiting configuration has been obtained from remote server or cached encrypted file
        while config_mem_cache.get_source() == "":
            sleep(1)
            if not is_all_threads_alive(thread_list): exit(1)
    else:
        config_mem_cache.set_all_conf(config_info)
        config_mem_cache.set_source("yarm_file")
    logger(f"Got configuration from {config_mem_cache.get_source()}")

    # print(f"config_mem_cache = {config_mem_cache.get_all_conf()}")

    # Create MQTT client for receiving raw data
    mqtt_conf=config_mem_cache.get_mqtt()
    mqtt_port=mqtt_conf.get("port")
    if mqtt_port is None: 
        logger("Invalid MQTT Port, Exit!")
        exit(1)
    # print("mqtt_conf=",mqtt_conf)
    unique_client_id = f"client_{uuid.uuid4()}"
    mqtt_client = mqtt.Client(client_id=unique_client_id, callback_api_version=CallbackAPIVersion.VERSION2)
    if mqtt_port > 8880:
            set_tls_psk_patch(mqtt_client, broker_psk_id, broker_psk)
    if "mqtt_user" in mqtt_conf:   # check if the broker requires username/password
        pwd = mqtt_conf.get("password", "")
        mqtt_client.username_pw_set(username=mqtt_conf["user"], password=pwd)
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect(mqtt_conf["ip"], mqtt_conf["port"], 30)
    mqtt_thread = threading.Thread(target=lambda: mqtt_client.loop_forever()) 
    mqtt_thread.daemon = True
    mqtt_thread.start()
    thread_list.append(mqtt_thread)

    # set up a connection to InfluxDB
    influxdb_conf=config_mem_cache.get_influxdb()
    # print("influxdb_conf=", influxdb_conf)
    url_str=influxdb_conf["ip"].split("://")
    if len(url_str)>=2:
        ssl_val = True if url_str[0] == "https" else False
        url_ip=url_str[1]
    else:
        ssl_val=False
        url_ip=url_str[0]
    verify=False    #ssl_val
    influxdb_client = InfluxDBClient(url_ip, influxdb_conf["port"], influxdb_conf["user"], influxdb_conf["password"], influxdb_conf["db"],ssl=ssl_val,verify_ssl=verify)
    msg_proc_thread=MsgProcThread(influxdb_client.write_points)
    thread_list.append(msg_proc_thread.get_thread())

    # Check the connection status
    try:
        if influxdb_client.ping():
            logger(f"Successfully connected to the Influxdb! {url_ip}")
            # print("")
        else:
            logger("Failed to connect to the Influxdb, exit")
            exit(1)
    except Exception as e:  
        logger("Ping failure, Connection to the Influxdb failed !, exit")
        exit(1)

    #Creating a thread pool for executing writes to influxDB
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # Using a buffer to collect data points, in order to reduce write operaton to InfluxDB
    influxdb_data_pionts_buf=[]
    last_save_time=time.monotonic()

    while True:
        #get result data message 
        msg_fetch=False
        try: 
            msg=msg_queue.get(timeout=1) # timeout 1 second  
            msg_fetch=True
        except queue.Empty:
            pass
        except Exception as e:  
            logger(f"Failed to get item from the queue: {e}")
        
        if msg_fetch:
            data_pionts = unpack_msg(msg)
            influxdb_data_pionts_buf +=data_pionts

        now=time.monotonic()
        # write data when number of data point reaches a threshold or time up
        number_of_piont=len(influxdb_data_pionts_buf)
        if (number_of_piont>0) and  (((now - last_save_time) >1) or number_of_piont>=5000) :
            last_save_time=now
            # print(influxdb_data_pionts_buf)

            # msg_proc_thread.put(influxdb_data_pionts_buf)

            data_to_write = influxdb_data_pionts_buf.copy()
            influxdb_data_pionts_buf.clear()
            #add a writing task to thread pool
            execute_asyn(influxdb_client.write_points,data_to_write)
            # influxdb_client.write_points(data_to_write)
            # if debug: print("Data Point Writed:", number_of_piont)

        # check if "thread" is still avlive, if no quit the program
        if not is_all_threads_alive(thread_list): break
        

    mqtt_client.disconnect()        
    influxdb_client.close()
    sleep(2)
    logger("========= mqtt_fwd_influx exit ===========")
    sys.exit()

