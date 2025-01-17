"""
Local API that routes the API requests to the correct server.

Functions in this file may
throw exceptions if the request fails. Make sure to catch these exceptions accordingly.
"""
import json
import os
import sys
import threading
from datetime import datetime
from time import sleep
import requests
import json
import socketio
import ssl
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from interface.csv_controller import CSVCtroller

# —— Constants (If not master server) —— #
IS_TLS = True
PORT = 3425
URL = 'ga.homedots.us'
#WEBSOCKET_URL and HTTP_URL will be modified in the initialization process of class Homedots.
WEBSOCKET_URL = f"{ 'https' if IS_TLS else 'http' }://{URL}:{PORT + 1}"
HTTP_URL = f"{ 'https' if IS_TLS else 'http' }://{URL}:{PORT}"
# SERVER_INSTANCE_NAME = 'test forwarder' # From mySQL forward_server_name or ai_server_name
# SERVER_INSTANCE_SECRET = 'sdgjhksdjh' # From mySQL
IS_MASTER_SERVER = os.environ.get('IS_MASTER_SERVER') == '1'

DEVICE_CONF_CSV= os.environ.get("LOCAL_DEVICE_CONF_CSV")
USER_CONF_CSV= os.environ.get("LOCAL_USER_CONF_CSV")
IS_LOCAL_CSV=USER_CONF_CSV and DEVICE_CONF_CSV

# —— Variables —— #
exit_flag = False

# —— Local Imports —— #
from interface import grafana_controller
from interface.utils import is_unix, is_mac

VITAL_MAPPING={'occupancy': ['vitals', 'occupancy'], \
               'heartrate': ['vitals', 'heartrate'], \
                'respiratoryrate': ['vitals', 'respiratoryrate'], \
                'systolic': ['vitals', 'systolic'],\
                'diastolic': ['vitals', 'diastolic'], \
                'quality': ['vitals', 'quality'], \
                'movement': ['vitals', 'movement'], \
                'alerts': ['vitals', 'alerts']}
# with open(os.path.join(os.path.dirname(__file__), "../../../config/config.json"), "r") as f:
#     VITAL_MAPPING = json.load(f)["vitalMapping"]

from interface.influxdb_controller import InfluxDBController
if IS_MASTER_SERVER:
    # from interface import grafana_controller
    from interface.mysql_controller import mysql_controller



# —— Exceptions —— #
class HTTPError(Exception):
    pass

class MissingFieldError(HTTPError):
    def __init__(self, msg = "Missing field in request"):
        super().__init__(msg)

class AuthorizationError(HTTPError):
    def __init__(self, msg = "Authorization credentials are incorrect"):
        super().__init__(msg)

class NotFoundError(HTTPError):
    def __init__(self, msg = "Resource not found"):
        super().__init__(msg)

class ParsingError(HTTPError):
    def __init__(self, msg = "Error parsing data"):
        super().__init__(msg)

def check_response(response):
    if response.status_code == 200:
        return

    msg = response.json()["message"] if "message" in response.json() else "Unknown error"
    if response.status_code == 400:
        raise MissingFieldError(msg)
    elif response.status_code == 401:
        raise AuthorizationError(msg)
    elif response.status_code == 404:
        raise NotFoundError(msg)
    else:
        raise HTTPError(msg)

# —— WebSocket Setup (Used only if not master) —— #
# def get_token(sio, server_instance_name, server_instance_secret, is_fwd):
#     global exit_flag
#     # Wait for the server to respond
#     sio.receive()

#     # Get the token
#     sio.emit('fwd' if is_fwd else 'ai', (server_instance_name, server_instance_secret))
#     res = sio.receive()[1]

#     if (res['msg'] != 'success'):
#         print('Error: ' + res['msg'])
#         sio.disconnect()
#         exit_flag = True
#         sys.exit(1)


#     return res['token']

# def get_token(sio, server_instance_name, server_instance_secret, is_fwd):
#     global exit_flag

#     try:
#         # Emit the event and wait for the server's response
#         event_name = 'fwd' if is_fwd else 'ai'
#         print(f"Sending event '{event_name}' with server_instance_name='{server_instance_name}'")

#         # Use sio.call with a timeout for synchronous response
#         response = sio.call(event_name, (server_instance_name, server_instance_secret), timeout=30)

#         # Check if response is None or if the response format is incorrect
#         if response is None:
#             print("Error: No response received from the server.")
#             sio.disconnect()
#             exit_flag = True
#             sys.exit(1)


#         # Check the response
#         print(f"Server response: {response}")
#         if 'msg' not in response or response['msg'] != 'success':
#             print('Error: Invalid response or failure message received.')
#             sio.disconnect()
#             exit_flag = True
#             sys.exit(1)

#         return response['token']

#     except Exception as e:
#         print(f"Error in get_token: {e}")
#         sio.disconnect()
#         exit_flag = True
#         sys.exit(1)



# def run_socket(server_instance_name, server_instance_secret, is_fwd, set_token):
#     global token, exit_flag
#     def target(server_instance_name, server_instance_secret, is_fwd, set_token):
#         global token, exit_flag

#         try:
#             with socketio.SimpleClient() as sio:
#                 sio.connect(WEBSOCKET_URL)

#                 while True:
#                     token = get_token(sio, server_instance_name, server_instance_secret, is_fwd)
#                     set_token(token)
#                     print("New token: Acquired")
#         except Exception as e:
#             print(e)
#             sio.disconnect()
#             exit_flag = True
#             sys.exit(1)
    
#     thread = threading.Thread(target=target, args=(server_instance_name, server_instance_secret, is_fwd, set_token))
#     thread.daemon = True
#     thread.start()


def get_token(sio, server_instance_name, server_instance_secret, is_fwd):
    global exit_flag
    
    try:
        event_name = 'fwd' if is_fwd else 'ai'
        # print(f"Sending event '{event_name}' with server_instance_name='{server_instance_name}'")
        
        # Emit the event and wait for the server's response
        try:
            response = sio.call(event_name, (server_instance_name, server_instance_secret), timeout=60)  # Increase timeout
        except Exception as e:
            print(f"sio.call Error: {e}")
        
        if response is None:
            print("Error: No response received from the server.")
            sio.disconnect()
            exit_flag = True
            sys.exit(1)

        # Log and verify the response
        # print(f"Server response: {response}")
        if 'msg' not in response or response['msg'] != 'success':
            print('Error: Invalid response or failure message received.')
            sio.disconnect()
            exit_flag = True
            sys.exit(1)

        return response['token']

    except Exception as e:
        print(f"Error in get_token: {e}")
        sio.disconnect()
        exit_flag = True
        sys.exit(1)


def run_socket(server_instance_name, server_instance_secret, is_fwd, set_token):
    global token, exit_flag

    def target(server_instance_name, server_instance_secret, is_fwd, set_token):
        global token, exit_flag

        try:
            # Initialize the Socket.IO client
            # print(f"sio.connect connecting {WEBSOCKET_URL}")
            http_session = requests.Session()
            http_session.verify = False
            sio = socketio.Client(http_session=http_session)

            @sio.event
            def connect():
                # print("I'm connected!")
                pass

            @sio.event
            def connect_error(data):
                # print("The connection failed!")
                pass

            @sio.event
            def disconnect(reason):
                # print(f"I'm disconnected! Reason:{reason}")
                pass

            # Connect to the WebSocket server
            sio.connect(WEBSOCKET_URL, transports=["websocket"])
            while True:
                # Replace `get_token` with your implementation
                token = get_token(sio, server_instance_name, server_instance_secret, is_fwd)
                set_token(token)
                print("New token: Acquired")
                break
        except Exception as e:
            print(f"Error: {e}")
            if 'sio' in locals():
                sio.disconnect()
            exit_flag = True
            sys.exit(1)

    # Start the WebSocket handling in a separate thread
    thread = threading.Thread(target=target, args=(server_instance_name, server_instance_secret, is_fwd, set_token))
    thread.daemon = True
    thread.start()

# —— Logic Layer —— #
class LogicLayer:
    def __init__(self):
        """
        Decides which server to route the request to
        """
        pass


# —— HomeDots API —— #
class HomeDots:
    def __init__(self, master_server, server_instance_name = "", server_instance_secret = "", is_fwd = True):
        """
        Initialize the HomeDots API

        This will start the WebSocket connection and set the token variable
        :raises: AuthorizationError if the token could not be retrieved
        """
        global exit_flag, WEBSOCKET_URL, HTTP_URL
        exit_flag=False

        WEBSOCKET_URL = f"{ 'https' if IS_TLS else 'http' }://{master_server}:{PORT + 1}"
        HTTP_URL = f"{ 'https' if IS_TLS else 'http' }://{master_server}:{PORT}"  

        self.token = ""
        self.server_instance_name = server_instance_name
        self.MASTER_SERVER_URL = HTTP_URL
        
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                self.csv_controller=CSVCtroller(USER_CONF_CSV, DEVICE_CONF_CSV)
                ip=os.environ.get("INFLUXDB_IP")
                port=os.environ.get("INFLUXDB_PORT")
                db=os.environ.get("INFLUXDB_DB")
                user=os.environ.get("INFLUXDB_USER")
                passwd=os.environ.get("INFLUXDB_PASSWD")
                self.csv_controller._set_influxdb_conf(ip, port, db, user, passwd)
            else:
                mysql_controller.connect()
        elif server_instance_name != "" or server_instance_secret != "":
            run_socket(server_instance_name, server_instance_secret, is_fwd, self.set_token)

            # Wait for the token to be set
            for _ in range(60):
                if self.token != "" or exit_flag is True:
                    break
                sleep(1)

            if self.token == "":
                raise AuthorizationError("Could not get token!")


# —— Setters —— #
    def set_token(self, token):
        self.token = token


# —— Authentication —— #
    def auth_user(self, user_name, user_token):
        """
        Authenticate a user
        :param user_name: The username
        :param user_token: The user token
        :return: (User ID, Bitstring representing user permissions) or False
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                return self.csv_controller.auth_user(user_name, user_token)
            else:
                return mysql_controller.auth_user(user_name, user_token)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/auth-user", json={
                "user_name": user_name
            }, headers={'Authorization': 'Bearer ' + user_token}, verify=False)

            if req.status_code == 200:
                return True
            else:
                return False


    def auth_admin(self, admin_name, admin_token):
        """
        Authenticate an admin
        :param admin_name: The admin username
        :param admin_token: The admin token
        :return: Admin ID or False
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                return self.csv_controller.auth_admin(admin_name, admin_token)
            else:
                return mysql_controller.auth_admin(admin_name, admin_token)
        else:
            # return True # Determinated by Master server
            req = requests.post(self.MASTER_SERVER_URL + "/auth-admin", json={
                "admin_name": admin_name
            }, headers={'Authorization': 'Bearer ' + admin_token}, verify=False)

            if req.status_code == 200:
                return True
            else:
                return False


    def auth_device(self, device_mac, device_token):
        """
        Authenticate a device
        :param device_mac: The device MAC address
        :param device_token: The device token
        :return: Device ID or False
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                return self.csv_controller.auth_device(device_mac, device_token)
            else:
                return mysql_controller.auth_device(device_mac, device_token)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/auth-device", json={
                "device_mac": device_mac,
            }, headers={'Authorization': 'Bearer ' + device_token}, verify=False)

            if req.status_code == 200:
                return True
            else:
                return False


    def auth_forward(self, forward_instance_name, forward_access_token):
        """
        Authenticate a forward server
        :param forward_instance_name: The forward server name
        :param forward_access_token: The forward access token
        :return: Forward server ID or False
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            return mysql_controller.auth_forward(forward_instance_name, forward_access_token)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/auth-forward", json={
                "forward_server_name": forward_instance_name,
            }, headers={'Authorization': 'Bearer ' + forward_access_token}, verify=False)

            if req.status_code == 200:
                return True
            else:
                return False


    def auth_ai(self, ai_instance_name, ai_access_token):
        """
        Authenticate an AI server
        :param ai_instance_name: The AI server name
        :param ai_access_token: The AI access token
        :return: AI server ID or False
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            return mysql_controller.auth_ai(ai_instance_name, ai_access_token)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/auth-ai", json={
                "ai_server_name": ai_instance_name,
            }, headers={'Authorization': 'Bearer ' + ai_access_token}, verify=False)

            if req.status_code == 200:
                return True
            else:
                return False


# —— For 3rd party commercial users ——
    def read_vitals_by_device_and_time(
        self,
        device_mac,
        device_token,
        timestamp = None,
        start_timestamp = None,
        end_timestamp = None,
        time_interval = None,
        vital_names = None,
    ):
        """
        Read vitals data from influxdb
        :param device_mac: The device MAC address
        :param device_token: The device token
        :param start_timestamp: The start time in epoch time
        :param end_timestamp: The end time in epoch time
        :param time_interval: The interval in seconds
        :param vital_names: The measurements to read [e.g. heartrate, temperature]
        :return: The data points e.g. [{time: in RFC3339 Format, measurement1: value, measurement2: value, ...}, ...]
        :raises: Exception if the request fails
        """

        start = start_timestamp or timestamp or int(datetime.now().timestamp())
        end = end_timestamp or 1700000000

        if is_unix(start) is False or is_unix(end) is False:
            raise ParsingError("Timestamps must be in epoch time")
        elif not end_timestamp:
            end = start
            start = end - 10

        interval = time_interval or end-start

        if not is_mac(device_mac):
            raise ParsingError("Device MAC address is invalid")
        elif vital_names is None or isinstance(vital_names, list) is False or len(vital_names) == 0:
            raise MissingFieldError("vital_names is missing or empty")

        for vital in vital_names:
            if isinstance(vital, str) is False or vital not in VITAL_MAPPING.keys():
                raise ParsingError("vital_name must be one of: " + ', '.join(VITAL_MAPPING.keys()))

        table_name = VITAL_MAPPING[vital_names[0]][0]

        influx_credentials = self.read_influx_conf_by_device(device_mac, device_token)

        if influx_credentials is None:
            raise NotFoundError("Influx configuration not found")

        influx_db = InfluxDBController(
            influx_credentials['ip'],
            influx_credentials['port'],
            influx_credentials['user'],
            influx_credentials['password'],
            influx_credentials['db']
        )

        result = influx_db.read_vitals_by_device_and_time(
            vital_names, table_name, device_mac, start, end, interval
        )

        if result is None:
            raise NotFoundError("Data not found")

        return result


    # def read_vitals_by_device_and_time2(
    #     self,
    #     device_mac,
    #     device_token,
    #     timestamp = None,
    #     start_timestamp = None,
    #     end_timestamp = None,
    #     time_interval = None,
    #     vital_names = None,
    # ):
    #     """
    #     Read vitals data from influxdb
    #     :param device_mac: The device MAC address
    #     :param device_token: The device token
    #     :param start_timestamp: The start time in epoch time
    #     :param end_timestamp: The end time in epoch time
    #     :param time_interval: The interval in seconds
    #     :param vital_names: The measurements to read [e.g. heartrate, temperature]
    #     :return: The data points e.g. [{time: in RFC3339 Format, measurement1: value, measurement2: value, ...}, ...]
    #     :raises: Exception if the request fails
    #     """
    #     if IS_MASTER_SERVER:
    #         start = start_timestamp or timestamp or int(datetime.now().timestamp())
    #         end = end_timestamp or 1700000000

    #         if is_unix(start) is False or is_unix(end) is False:
    #             raise ParsingError("Timestamps must be in epoch time")
    #         elif not end_timestamp:
    #             end = start
    #             start = end - 10

    #         interval = time_interval or end-start

    #         if not is_mac(device_mac):
    #             raise ParsingError("Device MAC address is invalid")
    #         elif vital_names is None or isinstance(vital_names, list) is False or len(vital_names) == 0:
    #             raise MissingFieldError("vital_names is missing or empty")

    #         for vital in vital_names:
    #             if isinstance(vital, str) is False or vital not in VITAL_MAPPING.keys():
    #                 raise ParsingError("vital_name must be one of: " + ', '.join(VITAL_MAPPING.keys()))

    #         table_name = VITAL_MAPPING[vital_names[0]][0]

    #         influx_credentials = self.read_influx_conf_by_device(device_mac, device_token)

    #         if influx_credentials is None:
    #             raise NotFoundError("Influx configuration not found")

    #         influx_db = InfluxDBController(
    #             influx_credentials['ip'],
    #             influx_credentials['port'],
    #             influx_credentials['user'],
    #             influx_credentials['password'],
    #             influx_credentials['db']
    #         )

    #         result = influx_db.read_vitals_by_device_and_time(
    #             vital_names, table_name, device_mac, start, end, interval
    #         )

    #         if result is None:
    #             raise NotFoundError("Data not found")

    #         return result
    #     else:
    #         body = {
    #             "device_mac": device_mac,
    #         }
    #         if timestamp is not None:
    #             body["timestamp"] = timestamp
    #         if start_timestamp is not None:
    #             body["start_timestamp"] = start_timestamp
    #         if end_timestamp is not None:
    #             body["end_timestamp"] = end_timestamp
    #         if time_interval is not None:
    #             body["time_interval"] = time_interval
    #         if vital_names is not None:
    #             body["vital_names"] = vital_names

    #         req = requests.post(self.MASTER_SERVER_URL + "/read-vitals-by-device-and-time", json=body,
    #                             headers={'Authorization': 'Bearer ' + device_token})

    #         check_response(req)

    #         return req.json()


    def read_device_list_by_user(self, user_name, user_token):
        """
        Read device list by user
        :param user_name: The username
        :param user_token: The user token
        :return: The device list {device_mac1: device_token1, device_mac2: device_token2, ...}
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                return self.csv_controller.read_device_list_by_user(user_name)
            else:
                return mysql_controller.read_device_list_by_user(user_name)
        else:
            token = user_token[:]
            if " " not in user_token:   # the user_token does not include a key word of authorization, such as "'Bearer ' + token"
                token= 'Bearer ' + token

            body={"user_name": user_name}
            
            req = requests.post(self.MASTER_SERVER_URL + "/read-device-list-by-user", json=body, headers={'Authorization': token}, verify=False)

            check_response(req)

            return req.json()


# —— For AI Server —— #
    def read_device_list_by_ai(self, ai_instance_name = None, ai_access_token = None):
        """
        Read device list by AI server name
        :param ai_instance_name: The AI server name
        :param ai_access_token: The AI access token
        :return: The device list [device_mac1, device_mac2, ...]
        :raises: Exception if the request fails
        """
        ai_instance_name = ai_instance_name or self.server_instance_name
        ai_access_token = ai_access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.read_device_list_by_ai(ai_instance_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-device-list-by-ai", json={
                "ai_server_name": ai_instance_name
            }, headers={'Authorization': 'Bearer ' + ai_access_token}, verify=False)

            check_response(req)

            return req.json()


    def read_alert_list_by_ai(self, ai_instance_name = None, ai_access_token = None):
        """
        Read alert list by AI server name
        :return: The alert dictionary {mac1: {alert} or None, mac2: {alert} or None, ...}
        :raises: Exception if the request fails
        """
        ai_instance_name = ai_instance_name or self.server_instance_name
        ai_access_token = ai_access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.read_alert_list_by_ai(ai_instance_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-alert-list-by-ai", json={
                "ai_server_name": ai_instance_name
            }, headers={'Authorization': 'Bearer ' + ai_access_token}, verify=False)

            check_response(req)
            # return req.json()

            result=req.json()
            
        alerts={}
        # print(result)
        for mac, alert in result.items():
            # alert is a string retrieve from mySQL, need to be convert to a dictionary
            if alert:
                alerts[mac]=json.loads(alert)

        return alerts


# —— For Forward Server —— #
    def read_device_list_by_forward(self, forward_instance_name = None, forward_access_token = None):
        """
        Read device list by forward server name
        :return: The device list [device_mac1, device_mac2, ...]
        :raises: Exception if the request fails
        """
        forward_instance_name = forward_instance_name or self.server_instance_name
        forward_access_token = forward_access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.read_device_list_by_forward(forward_instance_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-device-list-by-forward", json={
                "forward_server_name": forward_instance_name
            }, headers={'Authorization': 'Bearer ' + forward_access_token}, verify=False)

            check_response(req)

            return req.json()


    def read_influx_conf_by_forward(self, forward_instance_name = None, forward_access_token = None):
        """
        Read InfluxDB configuration by forward server name
        :return: The InfluxDB configuration {ip, port, db_raw, db_result, user, password}
        :raises: Exception if the request fails
        """
        forward_instance_name = forward_instance_name or self.server_instance_name
        forward_access_token = forward_access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.read_influx_conf_by_forward(forward_instance_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-influx-conf-by-forward", json={
                "forward_server_name": forward_instance_name
            }, headers={'Authorization': 'Bearer ' + forward_access_token}, verify=False)
            check_response(req)

            return req.json()


    def read_mqtt_conf_by_forward(self, forward_instance_name = None, forward_access_token = None):
        """
        Read MQTT configuration by forward server name
        :return: The MQTT configuration {ip, port, user, password, certificate, topic_filter, measurement_mapping}
        :raises: Exception if the request fails
        """
        forward_instance_name = forward_instance_name or self.server_instance_name
        forward_access_token = forward_access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.read_mqtt_conf_by_forward(forward_instance_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-mqtt-conf-by-forward", json={
                "forward_server_name": forward_instance_name
            }, headers={'Authorization': 'Bearer ' + forward_access_token}, verify=False)

            check_response(req)

            return req.json()
    
    def read_mqtt_conf_by_ai(self, instance_name = None, access_token = None):
        """
        Read MQTT configuration by ai server name
        :return: The MQTT configuration {ip, port, user, password, certificate, topic_filter, measurement_mapping}
        :raises: Exception if the request fails
        """
        instance_name = instance_name or self.server_instance_name
        access_token = access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.read_mqtt_conf_by_ai(instance_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-mqtt-conf-by-ai", json={
                "ai_server_name": instance_name
            }, headers={'Authorization': 'Bearer ' + access_token}, verify=False)

            check_response(req)

            return req.json()


    def read_influx_conf_by_device(self, device_mac, token):
        """
        Read InfluxDB configuration by device (Used by read-vitals-by-device-and-time)
        :param device_mac: The device MAC address
        :return: The InfluxDB configuration {ip, port, db_raw, db_result, user, password}
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                return self.csv_controller.read_influx_conf_by_device(device_mac)
            else:
                return mysql_controller.read_influx_conf_by_device(device_mac)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-influx-conf-by-device", json={
                "device_mac": device_mac
            }, headers={'Authorization': 'Bearer ' + token}, verify=False)

            check_response(req)

            return req.json()


    def read_tboard_conf_by_device(self, device_mac):
        """
        Read TBoard configuration by device
        :param device_mac: The device MAC address
        :return: The TBoard configuration {ip, port, user, password}
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            return mysql_controller.read_tboard_conf_by_device(device_mac)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/read-tboard-conf-by-device", json={
                "device_mac": device_mac
            }, verify=False)

            check_response(req)

            return req.json()


    def write_last_seen_by_device(self, device_mac, timestamp, for_raw, forward_instance_name = None, forward_access_token = None):
        """
        Write last seen by device
        :param forward_instance_name: The forward server name
        :param forward_access_token: The forward access token
        :param device_mac: The device MAC address
        :param timestamp: The timestamp in epoch time
        :param for_raw: True if the data is raw or False if the data is result
        :return: None
        :raises: Exception if the request fails
        """
        forward_instance_name = forward_instance_name or self.server_instance_name
        forward_access_token = forward_access_token or self.token

        if IS_MASTER_SERVER:
            return mysql_controller.write_last_seen_by_device(device_mac, timestamp, for_raw)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/write-last-seen-by-device", json={
                "forward_server_name": forward_instance_name,
                "device_mac": device_mac,
                "timestamp": timestamp,
                "for_raw": for_raw
            }, headers={'Authorization': 'Bearer ' + forward_access_token}, verify=False)

            check_response(req)

            return req.json()


    def register_license_by_device(self, user_name, user_token, mac_list):
        """
        Register the license for a number of devices and sets the expiration date
        of each device to the user's expiration date. The device token is generated
        based on a md5 hash of the device MAC address and expiration date.
        :param user_name: The username
        :param user_token: The user token
        :param mac_list: The list of device MAC addresses
        :return: A dictionary of the form {device_mac: device_token}
        :raises: Exception if the request fails
        """
        if IS_MASTER_SERVER:
            return mysql_controller.register_license_by_device(user_name, mac_list)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/register-license-by-device", json={
                "user_name": user_name,
                "mac_list": mac_list
            }, headers={'Authorization': 'Bearer ' + user_token}, verify=False)

            check_response(req)

            return req.json()


    def check_license_by_device(self, user_name, user_token, mac_list):
        pass

    def get_grafana_info_by_org(self, admin_name, admin_token, org_name):
        if IS_MASTER_SERVER:
            if IS_LOCAL_CSV:
                return self.csv_controller.get_grafana_info_by_org(org_name)
            else:
                return mysql_controller.get_grafana_info_by_org(org_name)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/get-grafana-info-by-org", json={
                "admin_name": admin_name,
                "org_name": org_name
            }, headers = {'Authorization': 'Bearer ' +  admin_token}, verify=False)

            return req.json()


    def refresh_grafana_dashboards(self, admin_name, admin_token, org_name, force_reset):

        result=self.get_grafana_info_by_org(admin_name, admin_token, org_name)
        # print(f"result={result}")
        if result is None:
            return False

        grafana_controller.create_overview_dashboard(
                result["user_name"], result["user_token"], result["user_id"], org_name, result["device_result"], force_reset
            )

        return True
        # if IS_MASTER_SERVER:
        #     result = mysql_controller.get_grafana_info_by_org(org_name)

        #     if result is None:
        #         return False

        #     user_result, device_result = result

        #     org_id, user_name, user_token = user_result

        #     grafana_controller.create_overview_dashboard(
        #         user_name, user_token, org_id, org_name, device_result, force_reset
        #     )

        #     return True
        # else:
        #     req = requests.post(self.MASTER_SERVER_URL + "/refresh-grafana-dashboards", json={
        #         "admin_name": admin_name,
        #         "org_name": org_name,
        #         "force_reset": force_reset
        #     }, headers = {'Authorization': 'Bearer ' + admin_token})

        #     check_response(req)

        #     return req.json()


    def check_status_2(self, device_mac):
        """
        Back compatible license control mechanism
        :param device_mac: The device MAC address
        :return: The IP, user, and password or {status: "0"} if the device is not registered
        """
        if IS_MASTER_SERVER:
            return mysql_controller.check_status_2(device_mac)
        else:
            req = requests.post(self.MASTER_SERVER_URL + "/checkStatus2", json={
                "mac": device_mac
            }, verify=False)

            return req.json()


    def write_data_by_device(self, device_mac, device_token, data):
        pass


    def read_data_by_device(self, device_mac, device_token):
        pass

