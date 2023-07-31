import threading
import queue
import time

from utils import return_gpu_info 

def get_info( gpu_info ):    
    while True:
        if gpu_info.empty():   
            gpu_info.put(return_gpu_info())
        time.sleep(0.001)            
    
def print_counter( gpu_info ):
    
    counter = 0    
    while (counter<100):        
        if not gpu_info.empty():
            print( str(counter) + ' ' + str(gpu_info.get()) )
            counter = counter + 1
        time.sleep(0.001)
        
if __name__ == '__main__':
    
    gpu_info = queue.Queue()
    
    gpu_thread = threading.Thread( target=get_info, daemon=True, args=(gpu_info,) ) 
    training_thread = threading.Thread( target=print_counter, args=(gpu_info,) )

    # Start the threads
    gpu_thread.start()
    training_thread.start()

    # Optionally, wait for the threads to complete
    training_thread.join()
    # gpu_thread.join() # no need to join(wait to finish since the thread is daemon)
    


