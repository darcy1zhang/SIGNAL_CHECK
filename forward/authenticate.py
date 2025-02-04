import sys,os
import threading
from security import *
from common import *
from time import sleep

current_dir=os.path.abspath(__file__)
project_package_dir = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(project_package_dir)
from interface.local_api import HomeDots

cache_conf_file_key=b'eBZVJQySCzk1YbTmF2LdO5Oqdn272oXhyFhzxuSRnC0='
# You can use the following function to generate a new key
# key = Fernet.generate_key()
# print(key)

def get_config_from_encrypt_file(path_to_cache,instance_name):

    cache_file_name = os.path.join(path_to_cache ,f'.{instance_name}')
    config_info=decrypt_dict(cache_conf_file_key, cache_file_name)
    return config_info

class ConfigMemCache:
    def __init__(self):
        '''
        config={"mqtt":{}, "alert"={}, "devices"=(), "influxdb"={}}
        '''
        self.config = {}
        self.config["mqtt"]={}
        self.config["alert"]={}
        self.config["influxdb"]={}
        self.config["devices"]=[]
        self.source=""
        self.lock = threading.RLock()

    def set_mqtt(self,mqtt_conf):
        '''
        arg:
        mqtt_conf={ip, port, user, password , topic_filter}
            ,among them, the topic_filter=["/organization/mac_addr/top",...]
        '''
        with self.lock:
            self.config["mqtt"]=mqtt_conf.copy()

    def get_mqtt(self):
        with self.lock:
            mqtt_conf=self.config["mqtt"].copy()
        return mqtt_conf
    
    def set_influxdb(self,influxdb_conf):
        '''
        arg:
        influxdb_conf={ip, port, db_raw, user, password,ssl}
        '''
        with self.lock:
            self.config["influxdb"]=influxdb_conf.copy()
        # self.influxdb['ssl']= "https"
    def get_influxdb(self):
        with self.lock:
            influxdb_conf=self.config["influxdb"].copy()
        return influxdb_conf
    
    def set_alert(self,alert):
        with self.lock:
            self.config["alert"]=alert.copy()
        # self.influxdb['ssl']= "https"
    def get_alert(self):
        with self.lock:
            alert=self.config["alert"].copy()
        return alert
    def get_alert_setting_by_mac(self, mac):
        with self.lock:
            alert_setting=self.config["alert"].get(mac)
        return alert_setting
    
    def set_devices(self,dev_list):
        '''
        arg:
        dev_list=["mac address",...]
        '''
        with self.lock:
            self.config["devices"]=dev_list.copy()

    def get_devices(self):
        with self.lock:
            dev_list=self.config["devices"].copy()
        return dev_list
    
    def is_authentic_mac(self, mac_addr):
        ret = False
        with self.lock:
            for dev_info in self.config["devices"]:
                if isinstance(dev_info, list):  # for AI configuation
                    if (mac_addr in dev_info) or ("*" in dev_info) :
                        ret=True
                elif isinstance(dev_info, str): # for forwarder config
                    if (mac_addr == dev_info) or ("*" in self.config["devices"]) :
                        ret=True
        return ret
    
    def get_monitor_target_by_mac(self, mac_addr):
        ret = ""
        with self.lock:
            for dev_info in self.config["devices"]:
                if (mac_addr in dev_info):
                    ret=dev_info[1]
        return ret
    
    def set_source(self,source):
        '''
        arg:
        source="database/cached_file/yarm_file"
        '''
        with self.lock:
            self.source=source
        # self.influxdb['ssl']= "https"
    def get_source(self):
        with self.lock:
            source=self.source
        return source
    
    def set_all_conf(self,conf):
        with self.lock:
            self.config=conf.copy()
    def get_all_conf(self):
        with self.lock:
            conf=self.config.copy()
        return conf
    


config_mem_cache=ConfigMemCache()

def update_authentic_mac_cache(master_server, instance_name, instance_token, path_to_cache, instance_type):
    """
    Get the config info
    :return: the config info
    """
    
    nap=60
    home_dots=None
    load_conf_from_encrypted_file=False

    while home_dots is None:
        try:
            home_dots = HomeDots(master_server, instance_name, instance_token,instance_type=="forwarder")
            if instance_type == 'forwarder':
                influxdb_conf=home_dots.read_influx_conf_by_forward()
                mac_info=home_dots.read_device_list_by_forward()
                mqtt_conf=home_dots.read_mqtt_conf_by_forward()
            elif instance_type == 'ai':
                mac_info=home_dots.read_device_list_by_ai()
                mqtt_conf=home_dots.read_mqtt_conf_by_ai()
                alert_conf=home_dots.read_alert_list_by_ai()
                # print(alert_conf)
            else:
                 return
        except Exception as e:
            logger(f"Connection to the database server failed: {e}")
            if not load_conf_from_encrypted_file:
                try:
                    file_conf=get_config_from_encrypt_file(path_to_cache, instance_name)
                except Exception as e:
                    logger(f"Failed to obtain configration from a cache file. Terminated : {e}")
                    sys.exit("Please Retry or Check database related configuration")
                config_mem_cache.set_all_conf(file_conf)
                config_mem_cache.set_source("cached_file")
                logger(f"Using cached configuation")
                load_conf_from_encrypted_file=True
            sleep(30)
    
    config_mem_cache.set_mqtt(mqtt_conf)
    if instance_type=="forwarder":
        config_mem_cache.set_influxdb(influxdb_conf)
    elif instance_type=="ai":
        config_mem_cache.set_alert(alert_conf)

    config_mem_cache.set_devices(mac_info)
    config_mem_cache.set_source("database")
    logger(f"Using configuation from database")

    cache_file_name = os.path.join(path_to_cache, f'.{instance_name}')
    # print(config_mem_cache.get_all_conf())
    try:
        encrypt_and_save_dict(config_mem_cache.get_all_conf(), cache_conf_file_key, cache_file_name)
    except Exception as e:
        logger(f"Error occured while writing: {cache_file_name}")
        sys.exit()

    # mac_set=[]
    latest_authentic_mac=[]
    while True:
        # mac_set.clear()
        try:
            if instance_type == 'forwarder':
                mac_info=home_dots.read_device_list_by_forward()
            else:
                mac_info=home_dots.read_device_list_by_ai()
                alert_conf=home_dots.read_alert_list_by_ai()
                # print(f"===============alert_conf={alert_conf}")
                if alert_conf:
                    config_mem_cache.set_alert(alert_conf)

        except  Exception as e:
            logger(f"Failed to get device list from remote database: {e}")
            sleep(10)
            continue    
        
        # mac_set.update(mac_info)
        # configuration is updated only when we get the same result in two concecutive read
        if (sorted(mac_info) == sorted(latest_authentic_mac)):
            cached_macs=config_mem_cache.get_devices()
            # update config_mem_cache
            if (sorted(cached_macs) != sorted(mac_info)):
                config_mem_cache.set_devices(mac_info)
                # save configuration to the cached file
                try:
                    encrypt_and_save_dict(config_mem_cache.get_all_conf(), cache_conf_file_key, cache_file_name)
                except Exception as e:
                    logger(f"Error occured while writing: {cache_file_name}")

                # log any change, including added and deleted device
                for d in mac_info:
                    if d not in cached_macs: 
                        info=f"Device Added: {d} from MySQL"
                        logging.info(info)

                for d in cached_macs:
                    if d not in mac_info: 
                        info=f"Device Deleted: {d} from MySQL"
                        logging.info(info)
            nap=60
        else:
            nap=15

        latest_authentic_mac=mac_info
        sleep(nap)


def start_authenticate_thread(master_server, instance_name, instance_token, path_to_cache=os.getcwd(), instance_type="ai"):
    '''
    master_server: where the MySQL located
    instance_name:
    instance_token: instance_secret
    path_to_cache: path to save the cached configuration file
    '''

    authentication_thread = threading.Thread(target=update_authentic_mac_cache, args=(master_server, instance_name, instance_token, path_to_cache, instance_type) )
    authentication_thread.daemon = True
    authentication_thread.start()
    return authentication_thread