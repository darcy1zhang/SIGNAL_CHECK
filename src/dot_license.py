import os
import re
from datetime import datetime, date
import subprocess
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from data_store_manager import DataStoreManager

TPM_HANDLE='0x81018699'
TPM_POLICY="sha256:0,1,2,3,4,5,6,7"

def hex_to_pem(rsa_hex):
    rsa_bytes = bytes.fromhex(rsa_hex)
    public_numbers = rsa.RSAPublicNumbers(
        e=65537,
        n=int.from_bytes(rsa_bytes, "big")
    )
    public_key = public_numbers.public_key(backend=default_backend())
    pem_key = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem_key   #.decode('utf-8'

def aes_decrypt_data(encrypted_data_with_iv, key):
    """Decrypt data and return its original content.
    
    :param encrypted_data_with_iv: Data with IV prepended (should be in bytes).
    :param key: AES key used for decryption (should be in bytes).
    :return: Decrypted data.
    """
    # extract IV and encrypted data
    iv = encrypted_data_with_iv[:16]  # First 16 bytes are IV
    encrypted_data = encrypted_data_with_iv[16:]  # Remaining bytes are encrypted daa
    
    # Create Cipher object
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    
    # Create decryptor object
    decryptor = cipher.decryptor()
    
    # Decrypt data
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    return decrypted_data

def fixed_deobfuscate(data):
    key=b'\x95\xfd\x94F\xb3\xbd\x00\xb2\x10\tLs\x02\xae\xc7\x11\xa8m\xcb#LR\xdf\x0cS\x83\xaa\xc1\x14\x9d)\x14'
    return(aes_decrypt_data(data,key))

class DotLicense:
    def __init__(self, authorization_file="ai_com.bin.tpm", tpm_rsa_ctx_file=TPM_HANDLE):

        self.tpm_rsa_ctx_file=tpm_rsa_ctx_file
        self.tpm_public_key=''
        self.number_of_devices_authorized=0
        self.expiration_date=0

        # Retrieve license info from "authorization_file"
        license_info=DataStoreManager(authorization_file,True)
        fake_key=self.decrypt_with_tpm(license_info.get("secondary_key"))
        key=fixed_deobfuscate(fake_key)
        self.tpm_public_key=aes_decrypt_data(license_info.get("rsa_pem_data"), key)
        self.number_of_devices_authorized=int(aes_decrypt_data(license_info.get("number_of_devices"), key))

        expire_date_str=aes_decrypt_data(license_info.get("expiration_date"), key)
        expire_date_str=expire_date_str.decode('utf-8')
        self.expiration_date=datetime.strptime(expire_date_str, '%Y-%m-%d').date()

        # Get tpm_public_key from TPM
        
    def get_public_key_from_tpm(self):
        # Get tpm_public_key from TPM
        result = subprocess.run(
            ['sudo','tpm2_readpublic', '-c', self.tpm_rsa_ctx_file, '-f', 'pem' ],
            check=True,
            capture_output=True,
            text=True  # Use text=True for string output, or False for bytes
        )
        match = re.search(r"rsa:\s+([a-f0-9]+)", result.stdout)
        if match:
            return(hex_to_pem(match.group(1)))
            # print(f"self.tpm_public_key={self.tpm_public_key}")
        else:
            print(f"Retrieve Key Failed")
            return None
            

    def encrypt_data_with_rsa_public_key(self, data):
        """
        Encrypt data using an RSA public key.

        :param public_key_file: Path to the file containing the RSA public key in PEM format.
        :param data: Data to be encrypted (should be in bytes).
        :return: Encrypted data.
        """
        # Load the public key
        # with open(self.public_key_file, 'rb') as key_file:
        public_key = serialization.load_pem_public_key(
            self.tpm_public_key,
            backend=default_backend()
        )

        # Encrypt the data
            # encrypted_data = public_key.encrypt(data,padding.PKCS1v15())
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_data

    def decrypt_with_tpm(self, encrypted_data):
        # print(f"==========encrypted_data/file======:{encrypted_data}")
        try:
            # Start the policy session and prepare for decryption.
            subprocess.run(["sudo", "tpm2_startauthsession", "--policy-session", "-S", "session.ctx"], check=True)
            subprocess.run(["sudo", "tpm2_policypcr", "-S", "session.ctx", "-l", TPM_POLICY],  stdout=subprocess.DEVNULL, check=True)

            result = subprocess.run(
                ['sudo', 'tpm2_rsadecrypt', '-c', self.tpm_rsa_ctx_file, '-p', 'session:session.ctx', '-s', 'oaep'],
                input=encrypted_data,  # In-memory data
                check=True,
                capture_output=True,
                text=False
            )
            # End session
            # subprocess.run(["sudo", "tpm2_flushcontext", "session.ctx"], check=True)
            # subprocess.run(['sudo','rm', 'session.ctx'], check=True)
            # print(f"==========decrypted_data_file======:{result.stdout}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Decryption failed with exit status {e.returncode}")
            print(f"Standard Output: {e.stdout}")
            print(f"Standard Error: {e.stderr}")
            raise
        finally:
            # Cleanup: Always flush and remove session context
            try:
                subprocess.run(["sudo", "tpm2_flushcontext", "session.ctx"], check=True)
            except subprocess.CalledProcessError as flush_err:
                print(f"Warning: Failed to flush session context: {flush_err}")

            try:
                subprocess.run(['sudo', 'rm', 'session.ctx'], check=True)
            except subprocess.CalledProcessError as rm_err:
                print(f"Warning: Failed to remove session context file: {rm_err}")

    def runtime_verify(self):
        random_data = os.urandom(160)  # 160*8-bit AES key
        encrpted_data=self.encrypt_data_with_rsa_public_key(random_data)
        random_feedback=self.decrypt_with_tpm(encrpted_data)
        if (random_data==random_feedback) and (not self.is_expired()):
            return True
        else:
            return False
    
    def number_of_devices(self):
        return self.number_of_devices_authorized

    def is_expired(self):
        today = datetime.today().date()
        # today = date(2025, 12, 4)
        return True if today > self.expiration_date else False
    
    def get_expiration_date(self):
        return self.expiration_date
    


