# HyperKAN: Kolmogorov-Arnold Networks make Hyperspectral Image Classifiers Smarter
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

## First phase of experiments

### Experiment 1: MLP & KAN variations, no hidden layers and grid size (G) tuning

*Overall classification accuracy (OA) for the baseline neural network models and datasets, in percent*

| Model Name        | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston 18 | KSC   | Average   | Average gain |
|-------------------|---------|--------|---------|---------------|-----------|------------|-------|-----------|--------------|
| KAN (in, out) G20 | 92.29   | 98.26  | 91.91   | 76.79         | 94.15     | 84.47      | 87.75 | 89.37     | 3.88         |
| KAN (in, out) G19 | 92.30   | 98.26  | 91.96   | 76.79         | 94.36     | 84.43      | 88.04 | 89.44     | 3.95         |
| KAN (in, out) G18 | 92.49   | 98.26  | 91.95   | 77.03         | 94.19     | 84.85      | 87.98 | 89.53     | 4.04         |
| KAN (in, out) G17 | 92.47   | 98.26  | 91.99   | 79.08         | 93.93     | 84.38      | 87.67 | 89.68     | 4.19         |
| KAN (in, out) G16 | 92.57   | 98.28  | 92.06   | 79.17         | 94.32     | 85.14      | 87.98 | 89.93     | 4.44         |
| KAN (in, out) G15 | 92.60   | 98.28  | 92.14   | 78.95         | 94.28     | 85.63      | 87.50 | 89.91     | 4.42         |
| KAN (in, out) G14 | 92.73   | 98.27  | 92.19   | 76.61         | 94.19     | 85.36      | 88.04 | 89.62     | 4.13         |
| KAN (in, out) G13 | 92.64   | 98.28  | 92.13   | 77.04         | 94.02     | 85.77      | 87.75 | 89.66     | 4.17         |
| KAN (in, out) G12 | 91.97   | 98.13  | 91.54   | 75.98         | 93.41     | 85.27      | 86.68 | 88.99     | 3.5          |
| KAN (in, out) G11 | 92.77   | 98.28  | 92.19   | 79.37         | 93.63     | 85.65      | 87.43 | 89.90     | 4.41         |
| KAN (in, out) G10 | 92.05   | 98.12  | 91.50   | 76.14         | 93.37     | 85.21      | 86.55 | 88.99     | 3.5          |
| KAN (in, out) G9  | 92.16   | 98.14  | 91.53   | 75.93         | 93.46     | 85.09      | 86.36 | 88.95     | 3.46         |
| KAN (in, out) G8  | 93.27   | 98.49  | 92.00   | 78.74         | 95.23     | 87.11      | 87.27 | **90.30** | **4.81**     |
| KAN (in, out) G7  | 93.35   | 98.50  | 91.98   | 77.28         | 94.11     | 86.25      | 87.33 | 89.82     | 4.33         |
| KAN (in, out) G6  | 91.93   | 98.17  | 91.14   | 76.00         | 94.15     | 86.39      | 87.54 | 89.33     | 3.84         |
| KAN (in, out) G5  | 91.71   | 98.41  | 91.95   | 78.79         | 95.01     | 87.14      | 87.20 | 90.03     | 4.54         |
| KAN (in, out) G4  | 91.53   | 98.16  | 90.99   | 76.18         | 93.24     | 85.99      | 86.13 | 88.88     | 3.39         |
| KAN (in, out) G3  | 91.13   | 98.12  | 90.81   | 75.90         | 93.33     | 85.91      | 86.34 | 88.79     | 3.3          |
| KAN (in, out) G2  | 90.62   | 98.30  | 90.87   | 78.64         | 93.85     | 86.36      | 87.08 | 89.38     | 3.89         |
| MLP (in, out)     | 87.15   | 97.87  | 90.35   | 75.64         | 88.17     | 80.86      | 78.41 | 85.49     | -            |

### Experiment 2: MLP & KAN variations, hidden layer size and grid size (G) tuning

*Overall classification accuracy (OA) for the baseline neural network models and datasets, in percent*

| Model Name            | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston 18 | KSC   | Average   | Average gain |
|-----------------------|---------|--------|---------|---------------|-----------|------------|-------|-----------|--------------|
| KAN (in, 4, out) G8   | 93.13   | 98.05  | 88.74   | 73.49         | 92.24     | 88.83      | 85.39 | **88.55** | **9.41**     |
| KAN (in, 4, out) G5   | 92.87   | 97.98  | 87.97   | 73.28         | 91.94     | 88.71      | 84.01 | 88.10     | 8.96         |
| KAN (in, 4, out) G2   | 90.09   | 97.61  | 87.70   | 71.63         | 94.45     | 86.20      | 84.53 | 87.45     | 8.31         |
| MLP (in, 4, out)      | 86.22   | 97.10  | 79.22   | 66.34         | 73.40     | 83.63      | 68.08 | 79.14     | -            |
| KAN (in, 8, out) G8   | 94.25   | 98.23  | 91.36   | 76.97         | 92.59     | 88.86      | 87.60 | 89.98     | 2.95         |
| KAN (in, 8, out) G5   | 94.76   | 98.68  | 92.33   | 77.45         | 93.72     | 89.27      | 86.95 | **90.45** | **3.42**     |
| KAN (in, 8, out) G2   | 92.01   | 98.20  | 92.16   | 76.56         | 91.98     | 89.26      | 89.22 | 89.91     | 2.88         |
| MLP (in, 8, out)      | 89.77   | 97.89  | 90.54   | 73.31         | 89.12     | 88.35      | 80.26 | 87.03     | -            |
| KAN (in, 16, out) G8  | 94.36   | 98.18  | 91.99   | 77.89         | 96.14     | 90.15      | 88.53 | **91.03** | **2.42**     |
| KAN (in, 16, out) G5  | 94.15   | 98.67  | 92.57   | 77.61         | 93.37     | 90.58      | 88.02 | 90.71     | 2.1          |
| KAN (in, 16, out) G2  | 93.34   | 98.45  | 91.94   | 75.78         | 94.67     | 89.30      | 87.39 | 90.12     | 1.51         |
| MLP (in, 16, out)     | 93.32   | 98.13  | 91.34   | 77.80         | 88.65     | 89.16      | 81.88 | 88.61     | -            |
| KAN (in, 32, out) G8  | 94.39   | 98.49  | 92.19   | 81.15         | 94.67     | 89.94      | 89.14 | 91.42     | 2.11         |
| KAN (in, 32, out) G5  | 94.27   | 98.74  | 92.16   | 79.82         | 95.62     | 90.74      | 89.01 | **91.48** | **2.17**     |
| KAN (in, 32, out) G2  | 93.90   | 98.47  | 92.24   | 80.34         | 95.40     | 89.79      | 89.45 | 91.37     | 2.06         |
| MLP (in, 32, out)     | 93.76   | 98.34  | 91.77   | 79.30         | 91.33     | 89.33      | 81.35 | 89.31     | -            |
| KAN (in, 64, out) G8  | 94.82   | 98.53  | 91.15   | 82.29         | 93.24     | 88.94      | 88.72 | 91.09     | 1.69         |
| KAN (in, 64, out) G5  | 93.66   | 98.61  | 92.06   | 80.85         | 95.06     | 90.11      | 88.57 | **91.27** | **1.87**     |
| KAN (in, 64, out) G2  | 93.84   | 98.48  | 92.22   | 81.52         | 92.76     | 88.21      | 89.50 | 90.93     | 1.53         |
| MLP (in, 64, out)     | 93.70   | 98.18  | 91.69   | 79.55         | 90.68     | 89.48      | 82.53 | 89.40     | -            |
| KAN (in, 128, out) G8 | 93.90   | 98.35  | 92.03   | 80.15         | 93.02     | 87.74      | 85.48 | 90.09     | 0.18         |
| KAN (in, 128, out) G5 | 93.98   | 98.62  | 92.12   | 80.95         | 95.06     | 90.05      | 87.48 | **91.18** | **1.27**     |
| KAN (in, 128, out) G2 | 93.85   | 98.52  | 91.88   | 79.71         | 92.81     | 88.38      | 85.48 | 90.09     | 0.18         |
| MLP (in, 128, out)    | 93.87   | 98.18  | 92.21   | 79.95         | 92.20     | 89.73      | 83.23 | 89.91     | -            |


### Experiment 3: MLP & KAN variations, two hidden layers sizes and grid size (G) tuning

*Overall classification accuracy (OA) for the baseline neural network models and datasets, in percent*

| Model Name                 | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston 18 | KSC   | AVG       | Gain     |
|----------------------------|---------|--------|---------|---------------|-----------|------------|-------|-----------|----------|
| KAN (in, 4, 4, out) G5     | 93.42   | 97.93  | 90.74   | 72.80         | 95.06     | 89.05      | 80.02 | **88.43** | **9.21** |
| KAN (in, 4, 4, out) G2     | 92.36   | 97.84  | 90.21   | 72.74         | 93.72     | 88.57      | 82.89 | 88.33     | 9.11     |
| MLP (in, 4, 4, out)        | 89.79   | 96.78  | 78.94   | 60.91         | 72.75     | 87.78      | 67.64 | 79.22     | -        |
| KAN (in, 8, 8, out) G5     | 94.86   | 98.45  | 91.62   | 76.58         | 93.67     | 89.98      | 85.12 | **90.04** | **2.98** |
| KAN (in, 8, 8, out) G2     | 94.06   | 97.98  | 91.45   | 74.89         | 95.23     | 90.03      | 85.23 | 89.83     | 2.77     |
| MLP (in, 8, 8, out)        | 93.43   | 98.06  | 90.56   | 75.13         | 85.66     | 89.41      | 77.21 | 87.06     | -        |
| KAN (in, 16, 16, out) G5   | 94.71   | 98.35  | 92.45   | 79.99         | 94.67     | 90.63      | 85.41 | 90.88     | 2.26     |
| KAN (in, 16, 16, out) G2   | 95.02   | 98.22  | 91.65   | 80.31         | 95.92     | 90.09      | 86.32 | **91.07** | **2.45** |
| MLP (in, 16, 16, out)      | 92.68   | 97.12  | 91.41   | 75.40         | 95.66     | 86.12      | 82.01 | 88.62     | -        |
| KAN (in, 32, 32, out) G5   | 94.43   | 98.45  | 92.13   | 82.24         | 94.93     | 91.08      | 86.97 | 91.46     | 1.2      |
| KAN (in, 32, 32, out) G2   | 94.88   | 98.69  | 91.88   | 82.02         | 94.80     | 90.89      | 87.25 | **91.48** | **1.22** |
| MLP (in, 32, 32, out)      | 94.54   | 98.21  | 91.58   | 78.12         | 96.75     | 87.81      | 84.87 | 90.26     | -        |
| KAN (in, 64, 64, out) G5   | 95.50   | 98.50  | 92.95   | 82.98         | 95.32     | 91.36      | 87.22 | **91.97** | **0.93** |
| KAN (in, 64, 64, out) G2   | 95.00   | 98.62  | 91.93   | 81.15         | 94.02     | 90.99      | 88.47 | 91.45     | 0.41     |
| MLP (in, 64, 64, out)      | 95.43   | 98.27  | 91.78   | 82.62         | 93.07     | 90.36      | 85.79 | 91.04     | -        |
| KAN (in, 128, 128, out) G5 | 95.31   | 98.66  | 92.80   | 82.71         | 95.79     | 91.10      | 87.60 | 91.99     | 0.77     |
| KAN (in, 128, 128, out) G2 | 94.84   | 98.34  | 92.05   | 83.79         | 96.36     | 90.98      | 88.00 | **92.05** | **0.83** |
| MLP (in, 128, 128, out)    | 94.66   | 97.72  | 90.07   | 83.47         | 95.01     | 90.83      | 86.78 | 90.79     | -        |

### Experiment 4: MLP & KAN variations, batch normalisation with two hidden sizes tuning

*Overall classification accuracy (OA) for the baseline neural network models and datasets, in percent*

| Model Name                 | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston 18 | KSC       | Average   | Average gain |
|----------------------------|-----------|-----------|-----------|---------------|-----------|------------|-----------|-----------|--------------|
| MLP (in, 16, 16, out)      | 92.85     | 97.91     | 91.16     | 80.28         | 97.09     | 87.25      | 88.38     | 90.70     | 1.25         |
| KAN (in, 16, 16, out) G2   | 95.81     | 98.66     | 93.56     | 80.63         | 94.84     | 89.50      | 90.70     | **91.95** |              |
| MLP (in, 32, 32, out)      | 94.49     | 98.29     | 92.66     | 84.08         | 97.27     | 89.46      | 90.02     | 92.32     | 0.5          |
| KAN (in, 32, 32, out) G2   | 95.12     | 98.61     | 94.26     | 83.32         | 96.96     | 90.34      | 91.14     | **92.82** |              |
| MLP (in, 64, 64, out)      | 95.69     | 98.47     | 92.98     | 85.77         | 97.01     | 89.92      | 91.03     | 92.98     | 0.6          |
| KAN (in, 64, 64, out) G2   | 96.14     | 98.98     | 93.70     | **87.86**     | 96.53     | 91.24      | 90.61     | **93.58** |              |
| MLP (in, 128, 128, out)    | 95.92     | 98.65     | 93.53     | 86.35         | 97.53     | 90.07      | **91.73** | 93.39     | 0.52         |
| KAN (in, 128, 128, out) G2 | **96.34** | **99.03** | **94.22** | **87.01**     | **97.57** | **92.04**  | 91.22     | **93.91** |              |

## Second phase of experiments

### Experiment 5: The replacement of Feature Extraction and Classifier Blocks

*Overall classification accuracy (OA) for 1DCNN and datasets, in percent*

[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/)

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston 18 | KSC       | Average   | Average gain |
|------------|-----------|-----------|-----------|---------------|-----------|------------|-----------|-----------|--------------|
| Vanilla    | 95.17     | 98.20     | 91.93     | 85.82         | 92.63     | 91.32      | 84.87     | 91.42     | -            |
| KAN-FE     | 95.37     | 98.96     | 94.70     | 86.87         | 97.22     | 92.91      | 89.64     | 93.66     | 2.24         |
| KAN-head   | 95.30     | 98.85     | 94.41     | 87.10         | 98.87     | 92.42      | 89.90     | 93.83     | 2.41         |
| Full KAN   | **95.68** | **99.07** | **95.28** | **88.90**     | **96.88** | **93.63**  | **90.91** | **94.33** | **2.91**     |

*\* KAN-FE indicates the replacement of the feature extractor block with KAN* \
*\* KAN-head indicates the replacement of the classifier block with KAN* \
*\* Full KAN indicates the replacement of both blocks*

*Overall classification accuracy (OA) for 3DCNN He and datasets, in percent*

[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/)

| Model Name | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston 18 | KSC       | Average   | Average gain |
|------------|-----------|-----------|-----------|---------------|-----------|------------|-----------|-----------|--------------|
| Vanilla    | 98.05     | 97.39     | 93.03     | 84.45         | 90.40     | 91.98      | 88.87     | 92.02     | -            |
| KAN-FE     | 98.74     | 99.45     | 92.25     | 89.39         | 91.69     | 90.62      | 86.77     | 92.70     | 0.68         |
| KAN-head   | 98.71     | 99.34     | 96.65     | 92.77         | 95.15     | 95.25      | 92.35     | 95.74     | 3.72         |
| Full KAN   | **98.76** | **99.71** | **98.14** | **96.47**     | **97.04** | **95.82**  | **93.66** | **97.08** | **5.06**     |

*\* KAN-FE indicates the replacement of the feature extractor block with KAN* \
*\* KAN-head indicates the replacement of the classifier block with KAN* \
*\* Full KAN indicates the replacement of both blocks*

## Third phase of experiments

### Experiment 6: The replacement of all MLP and Convolutional layers

*Overall classification accuracy (OA) for the various neural network models and datasets, in percent*

[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/) \
[3DCNN by Luo et al](https://ieeexplore.ieee.org/document/8455251) \
[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/) \
[SSFTT by Sun et al](https://ieeexplore.ieee.org/document/9684381)

| Model Name                                                                             | PaviaU*   | PaviaC    | Salinas   | Indian Pines* | Houston13 | Houston 18 | KSC       | Average   | Average gain |
|----------------------------------------------------------------------------------------|-----------|-----------|-----------|---------------|-----------|------------|-----------|-----------|--------------|
| Convolutions of spectral features                                                      |           |           |           |               |           |            |           |           |              |
| 1DCNN                                                                                  | 95.17     | 98.20     | 91.93     | 85.82         | 92.63     | 91.32      | 84.87     | 91.42     | 2.92         |
| 1DCNN KAN                                                                              | **95.68** | **99.07** | **95.28** | **88.90**     | **96.88** | **93.63**  | **90.91** | **94.33** |              |
| Convolutions of spatial features (window size 3x3)                                     |           |           |           |               |           |            |           |           |              |
| 2DCNN                                                                                  | 97.93     | 99.03     | 93.10     | 86.56         | 93.78     | 93.19      | 84.72     | 92.61     | 2.44         |
| 2DCNN KAN                                                                              | **99.12** | **99.57** | **96.52** | **93.77**     | **95.97** | **94.26**  | **86.19** | **95.05** |              |
| Convolutions of spectral-spatial features (window size 3x3)                            |           |           |           |               |           |            |           |           |              |
| 3DCNN Luo                                                                              | 96.92     | 99.21     | 93.99     | 81.90         | 95.69     | 92.45      | 87.37     | 92.50     | 3.00         |
| 3DCNN Luo KAN                                                                          | **99.02** | **99.57** | **96.97** | **91.33**     | **96.77** | **94.16**  | **90.72** | **95.50** |              |
| Convolutions of spectral-spatial features (window size 7x7)                            |           |           |           |               |           |            |           |           |              |
| 3DCNN He                                                                               | 98.05     | 97.39     | 93.03     | 84.45         | 90.40     | 91.98      | 88.87     | 92.02     | 5.05         |
| 3DCNN He KAN                                                                           | **98.76** | **99.71** | **98.14** | **96.47**     | **97.04** | **95.82**  | **93.66** | **97.08** |              |
| NM3DCNN                                                                                | 99.33     | 99.57     | 96.78     | 90.20         | 94.41     | 95.53      | 86.61     | 94.63     | 2.57         |
| NM3DCNN KAN                                                                            | **99.52** | **99.75** | **98.01** | **95.39**     | **97.53** | **95.84**  | **94.40** | **97.20** |              |
| Convolutions of spectral-spatial features (window size 13x13), 30 principal components |           |           |           |               |           |            |           |           |              |
| SSFTT                                                                                  | 99.86     | 99.88     | 99.85     | 98.78         | 98.57     | 96.22      | 95.45     | 98.37     | 0.8271       |
| SSFTT KAN                                                                              | **99.92** | **99.93** | **99.97** | **99.24**     | **99.46** | **97.12**  | **98.76** | **99.20** |              |

*Weighted F1 for the various neural network models and datasets, in percent*

[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/) \
[3DCNN by Luo et al](https://ieeexplore.ieee.org/document/8455251) \
[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/) \
[SSFTT by Sun et al](https://ieeexplore.ieee.org/document/9684381)

| Model Name    | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston 18 | KSC        | Average    | Average gain |
|---------------|------------|------------|------------|---------------|------------|------------|------------|------------|--------------|
| 1DCNN         | 0.9524     | 0.9811     | 0.9165     | 0.8577        | 0.9262     | 0.9125     | 0.8469     | 0.9133     | 0.0295       |
| 1DCNN KAN     | **0.9566** | **0.9903** | **0.9525** | **0.8883**    | **0.9680** | **0.9356** | **0.9088** | **0.9428** |              |
| 2DCNN         | 0.9788     | 0.9898     | 0.9307     | 0.8640        | 0.9366     | 0.9311     | 0.8465     | 0.9253     | 0.2442       |
| 2DCNN KAN     | **0.9910** | **0.9951** | **0.9646** | **0.9353**    | **0.9581** | **0.9423** | **0.8606** | **0.9495** |              |
| 3DCNN Luo     | 0.9689     | 0.9914     | 0.9392     | 0.8185        | 0.9563     | 0.9243     | 0.8728     | 0.9244     | 0.0300       |
| 3DCNN Luo KAN | **0.9898** | **0.9949** | **0.9690** | **0.9130**    | **0.9672** | **0.9413** | **0.9067** | **0.9545** |              |
| 3DCNN He      | 0.9801     | 0.9737     | 0.9301     | 0.8442        | 0.9036     | 0.9195     | 0.8882     | 0.9199     | 0.0506       |
| 3DCNN He KAN  | **0.9874** | **0.9968** | **0.9813** | **0.9646**    | **0.9700** | **0.9577** | **0.9364** | **0.9706** |              |
| NM3DCNN       | 0.9929     | 0.9954     | 0.9676     | 0.9014        | 0.9437     | 0.9552     | 0.8660     | 0.9460     | 0.0258       |
| NM3DCNN KAN   | **0.9951** | **0.9973** | **0.9800** | **0.9537**    | **0.9751** | **0.9583** | **0.9438** | **0.9719** |              |
| SSFTT         | 0.9985     | 0.9987     | 0.9985     | 0.9877        | 0.9856     | 0.9621     | 0.9543     | 0.9836     | 0.008271     |
| SSFTT KAN     | **0.9990** | **0.9992** | **0.9996** | **0.9923**    | **0.9945** | **0.9712** | **0.9875** | **0.9919** |              |

# Preprint

The results shown above may differ from the published preprint. The latter will be updated soon.

https://arxiv.org/abs/2407.05278
