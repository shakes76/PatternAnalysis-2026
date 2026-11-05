## Resource profile

Inference was measured on the NVIDIA Tesla T4 GPU using the full graph, with 5 warm-up runs and 30 timed runs. Memory is PyTorch peak allocated CUDA memory.  The GCN used more inference time and CUDA memory in this measurement. 

| Model | Parameters | Mean inference latency | Peak CUDA memory |
|-------|-----------:|-----------------------:|-----------------:|
| MLP   | 8,516      | 0.317 ms               | 35.58 MB         |
| GCN   | 8,516      | 4.162 ms               | 222.05 MB        |

