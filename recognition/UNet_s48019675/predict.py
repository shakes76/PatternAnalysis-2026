

import train as tr
import torch
import matplotlib.pyplot as plt
import modules
import dataset

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def run_2d():
    torch.cuda.reset_peak_memory_stats()
    thresholds = tr.training_2d()
    if torch.cuda.is_available():
            max_mem = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )
    
    with open("training_log.txt", "a") as log_file:
        log_file.write(
            f"Peak GPU Memory: {max_mem:.2f} GB\n"
        )
    print("testing")
    tr.test_2d(thresholds)
    print("done")

def run_3d():
    print("3d training")
    torch.cuda.reset_peak_memory_stats()
    m = tr.model_3d()
    m.train_model()
    if torch.cuda.is_available():
            max_mem = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )
    
    with open("training_log3d.txt", "a") as log_file:
        log_file.write(
            f"Peak GPU Memory: {max_mem:.2f} GB\n"
        )
    print("3d testing")
    m.test_model()

     
if __name__ == "__main__":
    run_2d()
    run_3d()

    
