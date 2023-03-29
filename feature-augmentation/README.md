# Efficient Graph Similarity Computation with More Expressive Power (feature augmentation)

The feature-augmentation method is developed based on [Efficient_Graph_Similarity_Computation](https://github.com/canqin001/Efficient_Graph_Similarity_Computation) and [Identity-aware Graph Neural Networks - ID-GNN-Fast](https://snap.stanford.edu/idgnn/)

## Training and Evaluation
[1. Train & Test with Teacher Model](https://github.com/hyllbd/Efficient_Graph_Similarity_Computation/blob/main/feature-augmentation/EGSC-T/README.md)

[2. Train & Test with KD-Student Model](https://github.com/hyllbd/Efficient_Graph_Similarity_Computation/blob/main/feature-augmentation/EGSC-KD/README.md)

## Dataset

```  AIDS700nef:  ``` https://drive.google.com/uc?export=download&id=10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z

```  LINUX:  ``` https://drive.google.com/uc?export=download&id=1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI

```  ALKANE:  ``` https://drive.google.com/uc?export=download&id=1-LmxaWW3KulLh00YqscVEflbqr0g4cXt

```  IMDBMulti:  ``` https://drive.google.com/uc?export=download&id=12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST


## k-length closed-circle feature augmentation related settings:

pleaes go to the folder EGSC-T and the folder EGSC-KD to train the teacher model and student model (through knowledge distillation) seperately.


```
--feature-aug 1 FCE (fast Closed-Circle Existence) approach
```

```
--feature-aug 2 FCC (fast Closed-Circle Counting) approach
```

```
--feature-aug 3 RCC (Real Close-Circle Existence) approach
```

```
--feature-aug 4 RCE (Real Closed-Circle Counting) approach
```

## Requirements
The codebase is implemented in Python 3.6.12. package versions used for development are just below.
```
matplotlib        3.3.4
networkx          2.4
numpy             1.19.5
pandas            1.1.2
scikit-learn      0.23.2
scipy             1.4.1
texttable         1.6.3
torch             1.6.0
torch-cluster     1.5.9
torch-geometric   1.7.0
torch-scatter     2.0.6
torch-sparse      0.6.9
tqdm              4.60.0
```

## Hint
On some datasets, the results are not quite stable. We suggest to run multiple times to report the avarage one.
