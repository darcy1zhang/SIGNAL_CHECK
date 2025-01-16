import numpy as np
from utils import *
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

good_data = []
hr_of_bsg_abp_label = []

def setup_mqtt_receiver(broker, port, topic):

    def on_message(client, userdata, msg):
        data = np.frombuffer(msg.payload, dtype=np.float64)
        # print(data.shape)
        bsg = data[:1000]
        abp = data[1000:2280]
        label = data[2280:]
        flag, hr_bsg = quality_control(bsg, 100)
        hr_label = np.median(label[-52:-42])
        hr_abp = get_hr_from_abp(abp)
        # print(f"Quality: {flag}, HR: {hr}")
        if flag:
            good_data.append(data)
            hr_of_bsg_abp_label.append([hr_bsg, hr_abp, hr_label])

    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_message = on_message
    client.connect(broker, port)
    client.subscribe(topic)
    client.loop_start()
    print("Receiver has started.")
    return client

def setup_mqtt_publisher(broker, port):

    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.connect(broker, port)
    print("Publisher has started.")
    return client

if __name__ == "__main__":
    
    mqtt_receiver = setup_mqtt_receiver("localhost", 1883, "test/topic")
    mqtt_publisher = setup_mqtt_publisher("localhost", 1883)

    # data_96 = np.load('./data/ID_96_10s_BSG_Art_Labels.npy')
    # data_98 = np.load('./data/ID_98_10s_BSG_Art_Labels.npy')
    data_105 = np.load('./data/ID_105_10s_BSG_Art_Labels.npy')
    # data_123 = np.load('./data/ID_123_10s_BSG_Art_Labels.npy')
    # data = np.vstack([data_96, data_98, dada_105, data_123])
    data = data_105
    for i in range(data.shape[0]):
        data_row = data[i]
        mqtt_publisher.publish("test/topic", data_row.tobytes())

    mqtt_publisher.disconnect()
    print("Publisher has stopped.")
    mqtt_receiver.loop_stop()
    mqtt_receiver.disconnect()
    print("Receiver has stopped.")

    good_data = np.array(good_data)
    hr_of_bsg_abp_label = np.array(hr_of_bsg_abp_label)
    plt.scatter(range(hr_of_bsg_abp_label.shape[0]), hr_of_bsg_abp_label[:, 0], s=1)
    plt.scatter(range(hr_of_bsg_abp_label.shape[0]), hr_of_bsg_abp_label[:, 1], s=1)
    plt.scatter(range(hr_of_bsg_abp_label.shape[0]), hr_of_bsg_abp_label[:, 2], s=1)
    plt.show()
    print(good_data.shape, data.shape, hr_of_bsg_abp_label.shape)
