# Efficient Graph Similarity Computation - Feature Augmentation - Teacher Model

## Train & Test
If you run the experiment on AIDS with FCE, then
```
python src/main.py --dataset AIDS700nef --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001 --feature-aug 1
```
If you run the experiment on LINUX with FCE, then
```
python src/main.py --dataset LINUX --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001 --feature-aug 1
```
If you run the experiment on IMDB with FCE, then
```
python src/main.py --dataset IMDBMulti --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001 --feature-aug 1
```
If you run the experiment on ALKANE with FCE, then
```
python src/main.py --dataset ALKANE --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001 --feature-aug 1
```


  
    
    
<b>You can replace the value of --feature-aug to switch different feature augmentation methods.</b>


```
--feature-aug 1 FCE (fast Closed-Circle Existence) appraoch
```

```
--feature-aug 2 FCC (fast Closed-Circle Counting) appraoch
```

```
--feature-aug 3 RCC (Real Close-Circle Existence) appraoch
```

```
--feature-aug 4 RCE (Real Closed-Circle Counting) appraoch
```
