from utils import tab_printer
from egsc_kd import EGSC_KD_Trainer
from parser import parameter_parser
import wandb
import torch
import numpy as np
import random

def main():
    args = parameter_parser()
    tab_printer(args)
    if args.wandb:
        wandb.init(config=args, project="Efficient_Graph_Similarity_Computation_EGSC-KD", settings=wandb.Settings(start_method="fork"))
    
    setup_seed(20)
    
    trainer = EGSC_KD_Trainer(args)
    trainer.load_model()
    trainer.fit()
    trainer.score()
    
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
    np.random.seed(seed) # numpy
    random.seed(seed) # python
    torch.manual_seed(seed) # cpu
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
