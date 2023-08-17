# Examining Effects of Gradient Accumulation and Gradient Checkpointing on GPU Memory Usage, GPU Utilization, and Training Time

Gradient Accumulation and Gradient Checkpointing are among the methods used to limit GPU memory usage during training.


In order to take advantage of these methods, one needs to sacrifice training time.


This project investigates the effects of Gradient Accumulation and Gradient Checkpointing on different parameters.


MNIST dataset is used as input and the size of the input images is increased to pressure GPU memory capacity.


The accuracy of the models is not presented since it is out of the scope of the project.


The project is implemented with Pytorch 2.0.1 and all of the source code and results are shared.


The run is performed in 2 parallel threads, while one of the threads performs training the other thread observes GPU status.


4 data loaders are used to feed GPUs with input data.


<!-- Blank line -->
<!-- Blank line -->

## --Without Gradient Checkpointing------With Gradient Checkpointing------

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_mem_usageADAM.png" width="400" />  
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_mem_usageADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_mem_usageADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_mem_usageADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_utilADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_utilADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_utilADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_utilADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_112x112_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_112x112_ADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_112x112_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_112x112_ADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_224x224_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_224x224_ADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_224x224_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_224x224_ADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_448x448_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_448x448_ADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_448x448_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_448x448_ADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_896x896_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_896x896_ADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_896x896_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_896x896_ADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_1792x1792_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_1792x1792_ADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_1792x1792_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_1792x1792_ADAM.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_3584x3584_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_3584x3584_ADAM.png" width="400" />  
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_3584x3584_ADAM.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_3584x3584_ADAM.png" width="400" />
</p>

<!-- Blank line -->
<!-- Blank line -->
<!-- Blank line -->

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_mem_usageSGD.png" width="400" />  
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_mem_usageSGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_mem_usageSGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_mem_usageSGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_utilSGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/gpu_utilSGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_utilSGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/gpu_utilSGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_112x112_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_112x112_SGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_112x112_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_112x112_SGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_224x224_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_224x224_SGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_224x224_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_224x224_SGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_448x448_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_448x448_SGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_448x448_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_448x448_SGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_896x896_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_896x896_SGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_896x896_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_896x896_SGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_1792x1792_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_1792x1792_SGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_1792x1792_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_1792x1792_SGD.png" width="400" />
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_3584x3584_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 2070/training_time_3584x3584_SGD.png" width="400" /> 
</p>

<p float="left">
  <img src="/00_github_media/dis_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_3584x3584_SGD.png" width="400" />
  <img src="/00_github_media/en_grad_checkpointing/NVIDIA GeForce RTX 3070 Ti/training_time_3584x3584_SGD.png" width="400" />
</p>






