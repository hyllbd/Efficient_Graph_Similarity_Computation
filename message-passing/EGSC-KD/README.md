# Efficient Graph Similarity Computation - KD Model

<!-- ![EGSC-KD](../Figs/KD.png) -->

## Train & Test with KD
If you run the experiment on AIDS, then
```
python src/main_kd.py --dataset AIDS700nef --gnn-operator ginmp --gnn-operator-fix ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000 --checkpoint path_to_checkpoint
```
If you run the experiment on LINUX, then
```
python src/main_kd.py --dataset LINUX --gnn-operator ginmp --gnn-operator-fix ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000 --checkpoint path_to_checkpoint
```
If you run the experiment on IMDB, then
```
python src/main_kd.py --dataset IMDB --gnn-operator ginmp --gnn-operator-fix ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000 --checkpoint path_to_checkpoint
```
If you run the experiment on ALKANE, then
```
python src/main_kd.py --dataset ALKANE --gnn-operator ginmp --gnn-operator-fix ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000 --checkpoint path_to_checkpoint
```