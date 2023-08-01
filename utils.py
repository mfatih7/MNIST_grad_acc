import torch
import os
import subprocess
import time
    
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

if __name__ == '__main__':
    while True:
        print_gpu_info()
        time.sleep(1)


