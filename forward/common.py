import concurrent.futures
import yaml
import threading
import queue

debug=False

#=================logging and config ========================
import logging

def setup_logger(log_filename):
    logging.basicConfig(
        filename=log_filename,  # 日志文件名
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )

def logger(info):
    logging.info(info)
    print(info)


def get_config_info_from_file(config_file):
    config_dict = {}
    with open(config_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict
#================== Thread pool ===========================================
#Creating a thread pool for executing writes to influxDB
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def execute_asyn_done_cb(fut):
    try:
        result = fut.result(timeout=10)
        # if debug: print("Write dateabase successful (callback):", result)
    except Exception as e:
        logger(f'Exception:{e}, execute_asyn_done_cb')

def execute_asyn(func,param):
    try:
        # summit function to the executor
        future = executor.submit(func, param)
        future.add_done_callback(execute_asyn_done_cb)

    except concurrent.futures.ThreadPoolExecutor as e:
        logger(f"ThreadPoolExecutor error: {e}")

#============== Thread 2=====================
class MsgProcThread:
    '''
    Creating a thread and a queue for a function to process messages asynchronously.

    '''
    def __init__(self, target_func, clear_msg=True, msg_maxsize=1000):
        '''
        target_func(msg)
        clear_msg: When setting True, after completing the precessing of message, the message will be cleared in order to release 
                   memory if it is a reference of "dict","list", or "set"
        '''
        self.clear_msg=clear_msg
        self.msg_queue=queue.Queue(maxsize=msg_maxsize)
        self.msg_proc_thread = threading.Thread(target=self.loop, args=(target_func,))
        self.msg_proc_thread.daemon = True
        self.msg_proc_thread.start()

    def loop(self, target_func):
        while True:
            try:
                # Get a message from the queue
                msg = self.msg_queue.get()
                
                # Process the message using the target function
                ret=target_func(msg)
                if not ret:
                    print(f"{target_func} return value= {ret}")
                if debug:
                    print(f"MsgProcThread: elements of message: {len(msg)} were processed ")
                
                # Clear the message if the clear_msg flag is set
                if self.clear_msg:
                    if isinstance(msg, (list, dict, set)):
                        msg.clear()
                        if debug:
                            print(f"MsgProcThread: message clear()")
                        
            except Exception as e:
                # Handle any exceptions that occur during message processing
                if 'logger' in globals():
                    logger(f"MsgProcThread: An error occurred: {e}")
    
    def put(self, msg):
        try:
            self.msg_queue.put(msg)
            if debug:
                print(f"MsgProcThread msg_queue water level={self.msg_queue.qsize()}")
        except Exception as e:
            if 'logger' in globals():
                logger(f"MsgProcThread: Failed to message into the queue: {e}")


    def get_thread(self):
        return self.msg_proc_thread
