**输入尺寸：** 512*512

===== DATE =====
Tue Feb 24 16:43:29 CST 2026
===== PYTHON =====
/home/lcy/miniconda3/envs/railseg2/bin/python
Python 3.10.19
===== PIP =====
pip 26.0.1 from /home/lcy/miniconda3/envs/railseg2/lib/python3.10/site-packages/pip (python 3.10)
===== NVIDIA-SMI =====
Tue Feb 24 16:43:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.108                Driver Version: 581.83         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   47C    P5             13W /   20W |       0MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
===== NUMPY/TORCH =====
numpy: 1.26.4
torch: 2.4.1+cu124
cuda available: True
torch cuda: 12.4
gpu: NVIDIA GeForce RTX 3050 Laptop GPU
cudnn: 90100
===== MMSEG STACK =====
mmcv: 2.1.0
mmengine: 0.10.7
mmseg: 1.2.2