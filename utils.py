import torch
import subprocess
import time
# import gc
    
def print_gpu_info():
    
    cmd = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
    result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    gpu_info = [line.split(', ') for line in result.split('\n')[1:]]
    for info in gpu_info:
        gpu_index, memory_used, memory_total, gpu_utilization = info
        device_name = torch.cuda.get_device_name(int(gpu_index))
        memory_used = int(memory_used.split()[0])
        memory_total = int(memory_total.split()[0])
        gpu_utilization = int(gpu_utilization.split()[0])
        print(f"GPU {gpu_index}: Device {device_name} Memory Used: {memory_used:.2f} MB, Memory Total: {memory_total:.2f} MB, GPU Utilization: {gpu_utilization} %")

def return_gpu_info():
    
    cmd = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
    result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    gpu_info = [line.split(', ') for line in result.split('\n')[1:]]
    for info in gpu_info:
        gpu_index, memory_used, memory_total, gpu_utilization = info
        device_name = torch.cuda.get_device_name(int(gpu_index))
        memory_used = int(memory_used.split()[0])
        memory_total = int(memory_total.split()[0])
        gpu_utilization = int(gpu_utilization.split()[0])
    return ( gpu_index, device_name, memory_used, memory_total, gpu_utilization )

def get_info( event_start_read_GPU_info, queue_gpu_info ):    
    
    while True:
        event_start_read_GPU_info.wait()        
        queue_gpu_info.put(return_gpu_info())        
        event_start_read_GPU_info.clear()
        # print('GPU event is cleared')

def clear_gpu_mem_util(model, images, labels):
    # model = None
    # images = None
    # labels = None
    # gc.collect()
    torch.cuda.empty_cache()
    
def check_gpu_info_queue(queue_gpu_info):
    # Make sure that no more data remains in queue 
    while(True):
        if(queue_gpu_info.empty()):
            break
        else:
            _ = queue_gpu_info.get()

def trig_GPU_read( queue_gpu_info, event_start_read_GPU_info ):    
    if ( queue_gpu_info.empty() and (not event_start_read_GPU_info.is_set() ) ) :
        event_start_read_GPU_info.set()
        # print("GPU read event is set")            
        
def get_info_from_GPU_queue( queue_gpu_info, event_start_read_GPU_info):
    if not queue_gpu_info.empty() and (not event_start_read_GPU_info.is_set()):            
        # print("GPU info is read")
        queue_gpu_info_taken = queue_gpu_info.get()
        device_name = queue_gpu_info_taken[1]
        device_mem_cap = queue_gpu_info_taken[3]
        gpu_mem_usage = int(queue_gpu_info_taken[2])
        gpu_util = int(queue_gpu_info_taken[4])
        return(device_name, device_mem_cap, gpu_mem_usage, gpu_util)
    else:
        return None

if __name__ == '__main__':
    while True:
        print_gpu_info()
        time.sleep(1)


