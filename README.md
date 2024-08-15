# HyperKAN: Kolmogorov-Arnold Networks make Hyperspectral Image Classificators Smarter
The objective of this research is to replace classical neural network layers with Kolmogorov-Arnold layers for the 
classification of hyperspectral data.

A novel approach to enhancing the efficacy of hyperspectral classifiers is proposed.

Pretrained weights will be able soon...

# Demo

To get started:
1. install dependencies `pip3 install -r requirements.txt`
2. create `datasets` folder with datasets files or set desired paths in `datasets_config.py`
3. set up the training and test model and specify the dataset name in `demo_train_predict.py`
4. run `python3 demo_train_predict.py`

# Acknowledgements
We have been using basic implementations of architectures from the following repositories:

1) [DeepHyperX](https://github.com/nshaud/DeepHyperX)
2) [HSI_SSFTT](https://github.com/zgr6010/HSI_SSFTT)
3) [KAN layers](https://github.com/IvanDrokin/torch-conv-kan)

# Results

* `*` The training sample size for PaviaU and Indian Pines consisted of 20% samples per class, while for the other datasets 
it was 10% per class.

* `Full` postfix means that all model layers was replaced with KAN layers when possible. Otherwise, the replaced layer is specified.

* `NxN` means the size of first and second hidden layers.

### Experiment 1: MLP & KAN as baseline approaches

*Overall classification accuracy (OA) for the baseline neural network models and datasets, in percent*

**MLP and KAN models**

| Model Name | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC   | Average OA | Average OA gain |
|------------|---------|--------|---------|---------------|-----------|-----------|-------|------------|-----------------|
| MLP 16     | 92.68   | 97.12  | 91.41   | 75.40         | 95.66     | 86.12     | 82.01 | 88.62      |                 |
| KAN 16     | 95.02   | 98.22  | 91.65   | 80.31         | 95.92     | 90.09     | 86.32 | 91.07      | 2.45            |
| MLP 32     | 94.54   | 98.21  | 91.58   | 78.12         | 96.75     | 87.81     | 84.87 | 90.26      |                 |
| KAN 32     | 94.88   | 98.69  | 91.84   | 82.02         | 96.10     | 89.47     | 86.45 | 91.35      | 1.09            |
| MLP 64     | 95.43   | 98.27  | 91.78   | 82.62         | 97.27     | 90.36     | 85.79 | 91.64      |                 |
| KAN 64     | 94.92   | 98.62  | 92.26   | 83.13         | 96.66     | 90.67     | 88.47 | 92.10      | 0.46            |
| MLP 128    | 91.66   | 97.72  | 90.07   | 73.63         | 80.72     | 87.00     | 73.13 | 84.84      |                 |
| KAN 128    | 94.84   | 98.05  | 91.21   | 83.79         | 96.36     | 90.98     | 88.00 | 89.39      | 4.55            |

**MLP and KAN models with Batch Normalisation**

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| MLP 16 BN  | 92.85     | 97.91     | 91.16     | 80.28         | 97.09     | 87.25     | 88.38     | 90.70      |                 |
| KAN 16 BN  | 95.81     | 98.66     | 93.56     | 80.63         | 94.84     | 89.50     | 90.70     | 91.95      | 1.25            |
| MLP 32 BN  | 94.49     | 98.29     | 92.66     | 84.08         | 97.27     | 89.46     | 90.02     | 92.32      |                 |
| KAN 32 BN  | 95.12     | 98.61     | 94.26     | 83.32         | 96.96     | 90.34     | 91.14     | 92.82      | 0.5             |
| MLP 64 BN  | 95.69     | 98.47     | 92.98     | 85.77         | 97.01     | 89.92     | 91.03     | 92.98      |                 |
| KAN 64 BN  | 96.14     | 98.98     | 93.70     | **87.86**     | 96.53     | 91.24     | 90.61     | 93.58      | 0.6             |
| MLP 128 BN | 95.92     | 98.65     | 93.53     | 86.35         | 97.53     | 90.07     | **91.73** | 93.39      |                 |
| KAN 128 BN | **96.34** | **99.03** | **94.22** | 87.01         | **97.57** | **92.04** | 91.22     | **93.91**  | 0.52            |


### Experiment 2: The replacement of Feature Extraction and Classifier Blocks

*Overall classification accuracy (OA) for 1DCNN and datasets, in percent*

[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/)

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| Vanilla    | 95.17     | 98.20     | 91.93     | 85.82         | 92.63     | 91.32     | 84.87     | 91.42      |                 |
| KAN-FE     | 95.37     | 98.96     | 94.70     | 86.87         | 97.22     | 92.91     | 89.64     | 93.66      | 2.24            |
| KAN-Head   | 95.30     | 98.85     | 94.41     | 87.10         | 98.87     | 92.42     | 89.90     | 93.83      | 2.41            |
| Full KAN   | **95.68** | **99.07** | **95.28** | **88.90**     | **96.88** | **93.63** | **90.91** | **94.33**  | **2.91**        |

*Overall classification accuracy (OA) for 3DCNN He and datasets, in percent*

[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/)

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| Vanilla    | 98.05     | 97.39     | 99.03     | 84.45         | 90.40     | 91.98     | 88.87     | 92.02      |                 |
| KAN-FE     | 98.74     | 99.45     | 92.25     | 89.39         | 91.69     | 90.62     | 86.77     | 92.70      | 0.68            |
| KAN-Head   | 98.71     | 99.34     | 96.65     | 92.77         | 95.15     | 95.25     | 92.35     | 95.74      | 3.72            |
| Full KAN   | **98.76** | **99.71** | **98.14** | **96.47**     | **97.04** | **95.82** | **93.66** | **97.08**  | **5.06**        |

### Experiment 3: The replacement of all MLP and Convolutional layers

#### *Overall classification accuracy (OA) for the various neural network models and datasets, in percent*

**Convolutions of spectral features**

[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/)

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| 1DCNN      | 95.17     | 98.20     | 91.93     | 85.82         | 92.63     | 91.32     | 84.87     | 91.42      |                 |
| 1DCNN KAN  | **95.68** | **99.07** | **95.28** | **88.90**     | **96.88** | **93.63** | **93.63** | **94.33**  | 2.92            |

**Convolutions of spacial features (window size 3x3)**

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| 2DCNN      | 97.93     | 99.03     | 93.10     | 86.56         | 93.78     | 93.19     | 84.72     | 92.61      |                 |
| 2DCNN KAN  | **99.12** | **99.57** | **96.52** | **93.77**     | **95.97** | **94.26** | **86.19** | **95.05**  | 2.44            |

**Convolutions of spectral-spacial features (window size 3x3)**

[3DCNN by Luo et al](https://ieeexplore.ieee.org/document/8455251)

| Model Name    | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|---------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| 3DCNN Luo     | 96.92     | 99.21     | 93.99     | 81.90         | 95.69     | 92.45     | 87.37     | 92.50      |                 |
| 3DCNN Luo KAN | **99.02** | **99.57** | **96.97** | **91.33**     | **96.77** | **94.16** | **90.72** | **95.50**  | 3.00            |

**Convolutions of spectral-spacial features (window size 7x7)**

[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/)

| Model Name   | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|--------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| 3DCNN He     | 98.05     | 97.39     | 93.03     | 84.45         | 90.40     | 91.98     | 88.87     | 92.02      |                 |
| 3DCNN He KAN | **98.76** | **99.71** | **98.14** | **96.47**     | **97.04** | **95.82** | **93.66** | **97.08**  | 5.05            |
| NM3DCNN      | 99.33     | 99.57     | 96.78     | 90.20         | 94.41     | 95.53     | 86.61     | 94.63      |                 |
| NM3DCNN KAN  | **99.52** | **99.75** | **98.01** | **95.39**     | **97.53** | **95.84** | **94.40** | **97.20**  | 2.57            |

**Convolutions of spectral-spacial features (window size 13x13), 30 principal components**

[SSFTT by Sun et al](https://ieeexplore.ieee.org/document/9684381)

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC       | Average OA | Average OA gain |
|------------|-----------|-----------|-----------|---------------|-----------|-----------|-----------|------------|-----------------|
| SSFTT      | 99.86     | 99.88     | 99.85     | 98.78         | 98.57     | 96.22     | 95.45     | 98.37      |                 |
| SSFTT KAN  | **99.92** | **99.93** | **99.97** | **99.24**     | **99.46** | **97.12** | **98.76** | **99.20**  | 0.82            |

#### *Weighted F1 for the various neural network models and datasets, in percent*

**Convolutions of spectral features**

[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/)

| Model Name | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        | Average F1w | Average F1w gain |
|------------|------------|------------|------------|---------------|------------|------------|------------|-------------|------------------|
| 1DCNN      | 0.9524     | 0.9811     | 0.9165     | 0.8577        | 0.9262     | 0.9125     | 0.8469     | 0.9133      |                  |
| 1DCNN KAN  | **0.9566** | **0.9903** | **0.9525** | **0.8883**    | **0.9680** | **0.9356** | **0.9088** | **0.9428**  | 0.0295           |

**Convolutions of spacial features (window size 3x3)**

| Model Name | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        | Average F1w | Average F1w gain |
|------------|------------|------------|------------|---------------|------------|------------|------------|-------------|------------------|
| 2DCNN      | 0.9788     | 0.9898     | 0.9307     | 0.8640        | 0.9366     | 0.9311     | 0.8465     | 0.9253      |                  |
| 2DCNN KAN  | **0.9910** | **0.9951** | **0.9646** | **0.9353**    | **0.9581** | **0.9423** | **0.8606** | **0.8606**  | 0.2442           |

**Convolutions of spectral-spacial features (window size 3x3)**

[3DCNN by Luo et al](https://ieeexplore.ieee.org/document/8455251)

| Model Name    | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        | Average F1w | Average F1w gain |
|---------------|------------|------------|------------|---------------|------------|------------|------------|-------------|------------------|
| 3DCNN Luo     | 0.9689     | 0.9914     | 0.9392     | 0.8185        | 0.9563     | 0.9243     | 0.8728     | 0.9244      |                  |
| 3DCNN Luo KAN | **0.9898** | **0.9949** | **0.9690** | **0.9130**    | **0.9672** | **0.9413** | **0.9067** | **0.9545**  | 0.03             |

**Convolutions of spectral-spacial features (window size 7x7)**

[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/)

| Model Name   | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        | Average F1w | Average F1w gain |
|--------------|------------|------------|------------|---------------|------------|------------|------------|-------------|------------------|
| 3DCNN He     | 0.9801     | 0.9737     | 0.9301     | 0.8442        | 0.9036     | 0.9195     | 0.8882     | 0.9199      |                  |
| 3DCNN He KAN | **0.9874** | **0.9968** | **0.9813** | **0.9646**    | **0.9700** | **0.9577** | **0.9364** | **0.9706**  | 0.0506           |
| NM3DCNN      | 0.9929     | 0.9954     | 0.9676     | 0.9014        | 0.9437     | 0.9552     | 0.8660     | 0.9460      |                  |
| NM3DCNN KAN  | **0.9951** | **0.9973** | **0.9800** | **0.9537**    | **0.9751** | **0.9583** | **0.9438** | **0.9719**  | 0.0258           |

**Convolutions of spectral-spacial features (window size 13x13), 30 principal components**

[SSFTT by Sun et al](https://ieeexplore.ieee.org/document/9684381)

| Model Name | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        | Average F1w | Average F1w gain |
|------------|------------|------------|------------|---------------|------------|------------|------------|-------------|------------------|
| SSFTT He   | 0.9985     | 0.9987     | 0.9985     | 0.9877        | 0.9856     | 0.9621     | 0.9543     | 0.9836      | 0.008271         |
| SSFTT KAN  | **0.9990** | **0.9992** | **0.9996** | **0.9923**    | **0.9945** | **0.9712** | **0.9875** | **0.9919**  |                  |

# Preprint

The results shown above may differ from the published preprint. The latter will be updated soon.

https://arxiv.org/abs/2407.05278
