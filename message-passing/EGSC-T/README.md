# Efficient Graph Similarity Computation - Teacher Model

<!-- ![EGSC-T](../Figs/Teacher-Net.png) -->

## Train & Test
If you run the experiment on AIDS, then
```
python src/main.py --dataset AIDS700nef --gnn-operator ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000
```
If you run the experiment on LINUX, then
```
python src/main.py --dataset LINUX --gnn-operator ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000
```
If you run the experiment on IMDB, then
```
python src/main.py --dataset IMDB --gnn-operator ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000
```
If you run the experiment on ALKANE, then
```
python src/main.py --dataset ALKANE --gnn-operator ginmp --epochs 6000 --batch-size 128 --learning-rate 0.001 --wandb 0 --test-interval 1000
```
