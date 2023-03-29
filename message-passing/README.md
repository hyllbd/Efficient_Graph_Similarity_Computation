# Efficient Graph Similarity Computation with More Expressive Power (Message Passing)

The message-passing method is developed based on [Efficient_Graph_Similarity_Computation](https://github.com/canqin001/Efficient_Graph_Similarity_Computation) and [SMP](https://github.com/cvignac/SMP)

## Training and Evaluation
[1. Train & Test with Teacher Model](https://github.com/hyllbd/Efficient_Graph_Similarity_Computation/blob/main/message-passing/EGSC-T/README.md)

[2. Train & Test with Student Model](https://github.com/hyllbd/Efficient_Graph_Similarity_Computation/blob/main/message-passing/EGSC-KD/README.md)

## Dataset

```  AIDS700nef:  ``` https://drive.google.com/uc?export=download&id=10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z

```  LINUX:  ``` https://drive.google.com/uc?export=download&id=1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI

```  ALKANE:  ``` https://drive.google.com/uc?export=download&id=1-LmxaWW3KulLh00YqscVEflbqr0g4cXt

```  IMDBMulti:  ``` https://drive.google.com/uc?export=download&id=12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST


## Requirements
The codebase is implemented in Python 3.9.0. package versions used for development are just below.
```
certifi==2022.12.7
charset-normalizer==2.1.1
click==8.1.3
cycler==0.11.0
decorator==5.1.1
dgl==0.9.1
docker-pycreds==0.4.0
gitdb==4.0.10
GitPython==3.1.30
idna==3.4
Jinja2==3.1.2
joblib==1.2.0
kiwisolver==1.4.4
MarkupSafe==2.1.1
matplotlib==3.3.4
mkl-fft==1.3.1
mkl-random==1.2.2
mkl-service==2.4.0
networkx==2.4
pandas==1.5.2
pathtools==0.1.2
promise==2.3
protobuf==4.21.12
psutil==5.9.4
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2022.7
PyYAML==6.0
requests==2.28.1
scikit-learn==0.23.2
scipy==1.9.3
sentry-sdk==1.12.1
setproctitle==1.3.2
shortuuid==1.0.11
smmap==5.0.0
texttable==1.6.3
thop==0.1.1.post2209072238
threadpoolctl==3.1.0
torch==1.10.1
torch-geometric==2.2.0
torchaudio==0.10.1
torchvision==0.11.2
tqdm==4.60.0
urllib3==1.26.13
wandb==0.13.7
```

## File Structure
```
|-- Checkpoints
|   |-- G_EarlyFusion_Disentangle_AIDS700nef_gin_checkpoint.pth
|   |-- G_EarlyFusion_Disentangle_ALKANE_gin_checkpoint.pth
|   |-- G_EarlyFusion_Disentangle_IMDBMulti_gin_checkpoint.pth
|   `-- G_EarlyFusion_Disentangle_LINUX_gin_checkpoint.pth
|-- EGSC-KD
|   |-- model_saved
|   |-- README.md
|   |-- src
|   |   |-- egsc_kd.py
|   |   |-- egsc_nonkd.py
|   |   |-- ginskip.py
|   |   |-- layers.py
|   |   |-- main_kd.py
|   |   |-- main_nonkd.py
|   |   |-- model_kd_light.py
|   |   |-- model_kd.py
|   |   |-- mpnn.py
|   |   |-- parser.py
|   |   |-- trans_modules.py
|   |   `-- utils.py
|   |-- train_kd.sh
|   `-- train_nonkd.sh
|-- EGSC-T
|   |-- model_saved
|   |-- README.md
|   |-- src
|   |   |-- egsc.py
|   |   |-- ginskip.py
|   |   |-- layers.py
|   |   |-- main.py
|   |   |-- model.py
|   |   |-- mpnn.py
|   |   |-- parser.py
|   |   |-- __pycache__
|   |   `-- utils.py
|   `-- train.sh
|-- GSC_datasets
|   `-- AIDS700nef
|       |-- processed
|       `-- raw
`-- README.md
```

## Acknowledgement
We would like to thank the [Efficient_Graph_Similarity_Computation
](https://github.com/canqin001/Efficient_Graph_Similarity_Computation) and [SMP](https://github.com/cvignac/SMP) which we used for this implementation.

## Hint
On some datasets, the results are not quite stable. We suggest to run multiple times to report the avarage one.
