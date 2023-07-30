import torch

def get_gpu_utilization( device, cur_or_max ):
    
    if(device.type == 'cpu'):
        return 0
    else:
        # torch.cuda.synchronize( torch.cuda.current_device() )
        if(cur_or_max == 'cur'):
            return torch.cuda.memory_allocated( torch.cuda.current_device() ) / 1024 / 1024
        elif(cur_or_max == 'max'):
            return torch.cuda.max_memory_allocated( torch.cuda.current_device() ) / 1024 / 1024
        else:
            assert(0)

        
def get_mem_info( device ):
    
    if(device.type == 'cpu'):
        return (0, 0)
    else:
        # torch.cuda.synchronize( torch.cuda.current_device() )
        a, b = torch.cuda.mem_get_info( torch.cuda.current_device() )
        return ( a / 1024 / 1024, b / 1024 / 1024 )
