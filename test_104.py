import numpy as np
from utils import *
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from scipy.signal import resample


def setup_mqtt_publisher(broker, port):

    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.connect(broker, port)
    print("Publisher has started.")
    return client

def setup_mqtt_receiver(broker, port, topic):

    def on_message(client, userdata, msg):
        mac_addr, timestamp, data_interval, data = parse_beddot_data(msg)
        # if mac_addr == '74:4d:bd:89:2e:3c':
        print('mac:', mac_addr, 'timestamp:', timestamp, 'data_interval:', data_interval, 'data:', len(data))

    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_message = on_message
    client.connect(broker, port)
    client.subscribe(topic, qos=1)
    # client.loop_start()
    client.loop_forever()
    print("Receiver has started.")
    return client


if __name__ == "__main__":
    
    mqtt_publisher = setup_mqtt_publisher('167.20.180.210', 1883)
    data = np.load('./BSG_ECG.npy')[:5]
    for i in range(data.shape[0]):
        data_row = data[i]
        bsg = data_row[:1000]
        ecg = data_row[1000:3560]
        ecg = resample(ecg, 1000)
        mac_bsg = data_row[-54]
        mac_ecg = mac_bsg + 100
        mac_bsg = int_to_mac(mac_bsg)
        mac_ecg = int_to_mac(mac_ecg)
        timestamp = int(data_row[-53])
        for j in range(9):
            byte_bsg = encode_beddot_data(mac_bsg, timestamp + j, 10000, 0, bsg[j * 100:(j + 1) * 100])
            mqtt_publisher.publish(f'/yida/{mac_bsg}/bsg', byte_bsg)
            byte_ecg = encode_beddot_data(mac_ecg, timestamp + j, 10000, 1, ecg[j * 100:(j + 1) * 100])
            mqtt_publisher.publish(f'/yida/{mac_ecg}/ecg', byte_ecg)

    mqtt_publisher.disconnect()
    print("Publisher has stopped.")

    # mqtt_receiver = setup_mqtt_receiver('167.20.180.210', 1883, "/+/+/geophone")
    # mqtt_receiver.loop_stop()
    # mqtt_receiver.disconnect()
    # print("Receiver has stopped.")