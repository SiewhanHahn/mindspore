(MindSpore) [ma-user work]$npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                 Version: 23.0.rc3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 1     910A                | Warning       | 69.6        35                0    / 0             |
| 0                         | 0000:81:00.0  | 0           2681 / 15137      1    / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
(MindSpore) [ma-user work]$
(MindSpore) [ma-user work]$free -h
              total        used        free      shared  buff/cache   available
Mem:          755Gi        40Gi       710Gi       128Mi       4.1Gi       713Gi
Swap:            0B          0B          0B
(MindSpore) [ma-user work]$lscpu
Architecture:        aarch64
Byte Order:          Little Endian
CPU(s):              192
On-line CPU(s) list: 0-191
Thread(s) per core:  1
Core(s) per socket:  48
Socket(s):           4
NUMA node(s):        8
Vendor ID:           0x48
Model:               0
Stepping:            0x1
BogoMIPS:            200.00
L1d cache:           64K
L1i cache:           64K
L2 cache:            512K
L3 cache:            24576K
NUMA node0 CPU(s):   0-23
NUMA node1 CPU(s):   24-47
NUMA node2 CPU(s):   48-71
NUMA node3 CPU(s):   72-95
NUMA node4 CPU(s):   96-119
NUMA node5 CPU(s):   120-143
NUMA node6 CPU(s):   144-167
NUMA node7 CPU(s):   168-191
Flags:               fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma dcpop asimddp asimdfhm
(MindSpore) [ma-user work]$python -c "import mindspore as ms; print('MindSpore Version:', ms.__version__)"
MindSpore Version: 1.10.0
(MindSpore) [ma-user work]$python -c "import mindspore as ms; print('Available Device:', ms.get_context('device_target'))"
Available Device: Ascend
(MindSpore) [ma-user work]$python3 --version
Python 3.7.10
(MindSpore) [ma-user work]$
(MindSpore) [ma-user work]$cat << 'EOF' > test.py
> import numpy as np
> import mindspore as ms
> import mindspore.ops as ops
> 
> # 设置运行动静态图模式，指定后端为 Ascend
> ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
> 
> # 创建两个张量进行矩阵乘法
> x = ms.Tensor(np.ones([3, 3]), ms.float32)
> y = ms.Tensor(np.ones([3, 3]), ms.float32)
> output = ops.matmul(x, y)
> 
> print("矩阵计算结果:\n", output)
> EOF
(MindSpore) [ma-user work]$
(MindSpore) [ma-user work]$python test.py
矩阵计算结果:
 [[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
(MindSpore) [ma-user work]$