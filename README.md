# Efficient Graph Similarity Computation with More Expressive Power (L45 Project)


This repo contains the source code and dataset for our L45 mini project.

Authors: Wenzhao Li (wl301@cam.ac.uk) and Zejian Li (zl525@cam.ac.uk)


## Abstract
<div>
    <br>
Graph similarity computation (GSC) is a fundamental task that plays a crucial role in numerous practical applications such as drug design, anomaly detection, and identification of social groups. Complex connections and interactions between input graphs in GSC are inherently computationally expensive. Recent studies indicate that a teacher-student knowledge distillation framework can achieve both speed and accuracy. This framework enables the teacher model to perform slow learning of early feature fusion without impacting the reference speed of the student model. Given this advantage, we choose to investigate the possibility of enhancing the expressive power of GSC. Our objective is to design a more effective graph similarity computation framework with enhanced expressive power. We integrate k-length closed-circle feature augmentation and structural message-passing (SMP) into this framework separately. Experimental results indicate that our methods can improve the performance of GSC on several datasets. We present our observations and findings in the Experiments and Evaluation section.
    <br>
</div>

## Dataset

We have used the standard dataloader, i.e., ‘GEDDataset’, directly provided in the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ged_dataset.html#GEDDataset).

```  AIDS700nef:  ``` https://drive.google.com/uc?export=download&id=10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z

```  LINUX:  ``` https://drive.google.com/uc?export=download&id=1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI

```  ALKANE:  ``` https://drive.google.com/uc?export=download&id=1-LmxaWW3KulLh00YqscVEflbqr0g4cXt

```  IMDBMulti:  ``` https://drive.google.com/uc?export=download&id=12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST


## k-length closed-circle feature augmentation related settings:

pleaes go to the folder EGSC-T and the folder EGSC-KD for more training details.

```
--feature-aug -1: orig baseline with orign shuffle function
```

```
--feature-aug 0 (default): orig baseline with updated shuffle function
```

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

## Message-passing instructions:

When using message-passing for experiments, please refer to [Message-passing README](https://github.com/hyllbd/Efficient_Graph_Similarity_Computation/blob/main/message-passing/README.md)


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
