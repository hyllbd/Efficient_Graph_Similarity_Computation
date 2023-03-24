from utils import tab_printer
from egsc import EGSCTrainer
from parser import parameter_parser
import wandb
import torch
import numpy as np
import random

def main():
    # python src/main_kd.py --dataset AIDS700nef --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001
    args = parameter_parser()
    if args.wandb:
        wandb.init(config=args, project="Efficient_Graph_Similarity_Computation_EGSC-T", settings=wandb.Settings(start_method="fork"))
    tab_printer(args)
    setup_seed(20)
    trainer = EGSCTrainer(args)
    
    trainer.fit()
    trainer.score()
    trainer.save_model()
    
    if args.notify:
        import os
        import sys
        if sys.platform == 'linux':
            os.system('notify-send EGSC "Program is finished."')
        elif sys.platform == 'posix':
            os.system("""
                      osascript -e 'display notification "EGSC" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No notification support for this OS.')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
