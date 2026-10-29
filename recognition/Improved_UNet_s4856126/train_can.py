"""Convenience entry point: CAN training with the shared train.py pipeline.

Example:
  python train_can.py --smoke-test --device cpu
  python train_can.py --epochs 10 --batch-size 8 --device cuda
"""

import sys
import train


if __name__ == "__main__":
    # The main training script owns all training logic, checkpoints and outputs.
    # Use '--model can' so no existing Basic U-Net checkpoint is overwritten.
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "can"])
    train.main()
