import threading
import queue
import time


def get_time(q):
    while True:
        if q.empty():
            a = time.perf_counter()
            q.put(a)
            time.sleep(1)
    
def print_counter(q):
    
    for i in range(100):
        if not q.empty():
            print( str(i))
            local_time = q.get()
            print( local_time )
            time.sleep(3)
        
if __name__ == '__main__':
    
    gpu_memory_queue = queue.Queue()
    
    gpu_thread = threading.Thread(target=get_time, args=( gpu_memory_queue,) ) 
    training_thread = threading.Thread(target=print_counter, args=( gpu_memory_queue,) )

    # Start the threads
    gpu_thread.start()
    training_thread.start()

    # Optionally, wait for the threads to complete
    gpu_thread.join()
    training_thread.join()





    
    
    
    
