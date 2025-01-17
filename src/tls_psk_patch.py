"""
`paho mqtt client` using `TSL-PSK` communication.

Before using this module,Please install "libssl-dev" package and "sslpsk2" Python library:

sudo apt-get update
sudo apt-get install libssl-dev
pip3 install sslpsk2

Check the relevant details about the environment on the repository:
https://github.com/maovidal/paho_sslpsk2_demo 
[Reminder: In above demo, use SSLPSKContext(ssl.PROTOCOL_TLSv1_2) instead of "SSLPSKContext(ssl.PROTOCOL_TLS_CLIENT)"]
"""


import ssl
from sslpsk2.sslpsk2 import _ssl_set_psk_server_callback, _ssl_set_psk_client_callback
import string
# import paho.mqtt.client as mqtt

# Some definitions to be modified according to your setup:
# CONFIG_MQTT_BROKER='ga.homedots.us'
# CONFIG_MQTT_BROKER_PORT=8883
CONFIG_CLIENT_ID='paho_sslpsk2' # Having a fixed ID may help debugging on the broker
# CONFIG_IDENTITY=b'abc'
# CONFIG_HEX_PSK='123456'
# CONFIG_TOPIC_TO_SUBSCRIBE="hello"



# Modify SSLContext to use sslpsk2
# https://github.com/drbild/sslpsk/issues/19#issue-547462099

def _ssl_setup_psk_callbacks(sslobj):
    psk = sslobj.context.psk
    hint = sslobj.context.hint
    identity = sslobj.context.identity
    if psk:
        if sslobj.server_side:
            cb = psk if callable(psk) else lambda _identity: psk
            _ssl_set_psk_server_callback(sslobj, cb, hint)
        else:
            cb = psk if callable(psk) else lambda _hint: psk if isinstance(psk, tuple) else (psk, identity)
            _ssl_set_psk_client_callback(sslobj, cb)


class SSLPSKContext(ssl.SSLContext):
    @property
    def psk(self):
        return getattr(self, "_psk", None)

    @psk.setter
    def psk(self, psk):
        self._psk = psk

    @property
    def hint(self):
        return getattr(self, "_hint", None)

    @hint.setter
    def hint(self, hint):
        self._hint = hint

    @property
    def identity(self):
        return getattr(self, "_identity", None)

    @identity.setter
    def identity(self, identity):
        self._identity = identity


class SSLPSKObject(ssl.SSLObject):
    def do_handshake(self, *args, **kwargs):
        if not hasattr(self, '_did_psk_setup'):
            _ssl_setup_psk_callbacks(self)
            self._did_psk_setup = True
        super().do_handshake(*args, **kwargs)


class SSLPSKSocket(ssl.SSLSocket):
    def do_handshake(self, *args, **kwargs):
        if not hasattr(self, '_did_psk_setup'):
            _ssl_setup_psk_callbacks(self)
            self._did_psk_setup = True
        super().do_handshake(*args, **kwargs)


SSLPSKContext.sslobject_class = SSLPSKObject
SSLPSKContext.sslsocket_class = SSLPSKSocket


# Preparations to use the new SSLPSKContext object with Paho
# https://github.com/eclipse/paho.mqtt.python/issues/451#issuecomment-705623084

context = SSLPSKContext(ssl.PROTOCOL_TLS_CLIENT)
context = SSLPSKContext(ssl.PROTOCOL_TLSv1_2)
context.set_ciphers('PSK')
# context.psk = bytes.fromhex(CONFIG_HEX_PSK)
# context.identity = CONFIG_IDENTITY

def set_tls_psk_patch(mqtt_client, psk_id, psk):
    '''
    input: 
        mqtt_client: that is the return value of mqtt.Client()
        psk_id: PSK id, e.g. 'abc'
        psk: e.g. '123456'
    '''
    global context

    # context.set_ciphers('PSK')
    context.identity = psk_id.encode('utf-8')
    context.psk = bytes.fromhex(psk)
    mqtt_client.tls_set_context(context) 

    # print(f"Using OpenSSL version: {ssl.OPENSSL_VERSION}")
    return
