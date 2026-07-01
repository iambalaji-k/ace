import sys
import os
sys.path.append('.')

import torch
from scripts.training.train_self_play_v3 import collect_match_trajectories, AceNetV2

def main():
    print("Testing seed 450016...")
    model = AceNetV2()
    # Check if we can run it
    collect_match_trajectories(model, "checkpoints", 450016)
    print("Completed successfully!")

if __name__ == "__main__":
    main()
