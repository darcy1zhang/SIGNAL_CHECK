from cryptography.fernet import Fernet
import json
import os

def generate_key():
    key = Fernet.generate_key()
    print(key)

def encrypt_and_save_dict(my_dict, key, data_file='fwd_cache_cfg.bin'):
    """
    Encrypt a dictionary and save it to a file along with the encryption key.

    Args:
    - my_dict (dict): Dictionary to encrypt.
    - data_file (str): Filename to save the encrypted data.
    - key_file (str): Filename to save the encryption key.

    Returns:
    - None
    """
    # Generate a key
    # key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    # Convert dictionary to JSON string
    json_str = json.dumps(my_dict)

    # Encrypt the JSON string
    encrypted_data = cipher_suite.encrypt(json_str.encode())

    # Save the encrypted data and key to files
    directory = os.path.dirname(data_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(data_file, 'wb') as f:
        f.write(encrypted_data)

def decrypt_dict(key, data_file='fwd_cache_cfg.bin'):
    """
    Decrypt a file and restore it to a dictionary.

    Args:
    - data_file (str): Filename of the encrypted data.
    - key_file (str): Filename of the encryption key.

    Returns:
    - dict: Decrypted dictionary.
    """
    # Load the key
    # with open(key_file, 'rb') as f:
    #     key = f.read()

    cipher_suite = Fernet(key)

    # Load the encrypted data
    with open(data_file, 'rb') as f:
        encrypted_data = f.read()

    # Decrypt the data
    decrypted_data = cipher_suite.decrypt(encrypted_data)

    # Convert JSON string back to dictionary
    my_dict = json.loads(decrypted_data.decode())

    return my_dict

'''
# Example usage
my_dict = {"key1": "value1", "key2": "value2"}
encrypt_and_save_dict(my_dict)

# Later...
restored_dict = decrypt_dict()
print(restored_dict)
'''