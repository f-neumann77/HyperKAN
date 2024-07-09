# HyperKAN
Replacement classic layers with Kolmogorov-Arnold layers for Hyperspectral data classification

We propose new approach fro increasing hyperspectral classifiers performance

# Demo

For start train and test set model, dataset and params in _demo_train_predict.py_ and to execute `python3 demo_train_predict.py`

# Acknowledgements
We used basic implementations of architectures from repositories:

1) [DeepHyperX](https://github.com/nshaud/DeepHyperX)
2) [HSI_SSFTT](https://github.com/zgr6010/HSI_SSFTT)
3) [KAN layers](https://github.com/IvanDrokin/torch-conv-kan)

# Results

### Baseline 

| Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|MLP|	92.54 |	97.95 |	91.23 |	80.66	| 95.06	| 86.62 |	81.84 |
|KAN| 93.40 |	98.53	|92.28|	83.70	|95.40	|87.70	|88.65|

### Spectral-features

|Model|PaviaU*|PaviaC|Salinas|Indian Pines*|Houston13|Houston18|KSC|

|--------|--------|--------|--------|--------|--------|--------|--------|--------|

|1DCNN Hu|	94.21	|98.20	|91.93	|85.82	|92.63	|91.32	|84.87|

|1DCNN Hu + KAN-Head|	95.63 |	99.04	|94.95	|87.17	|96.10	|93.00	|88.95|

|1DCNN Hu full KAN|95.68|	99.07|	95.28|	88.90	|96.88|	93.63	|90.91|

M1DCNN	|94.79	|98.81	|91.95	|84.22	|93.89	|91.90	|85.67|

M1DCNN + KAN-Head |	95.26	|98.99|	92.83|	85.56|	96.75|	93.86|	91.12|


### Spectral-spatial features (window size 3x3)

|Model|PaviaU*|PaviaC|Salinas|Indian Pines*|Houston13|Houston18|KSC|
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
3DCNN Luo |	96.92 |	99.21 |	93.99	|81.90	|95.69	|92.45	|87.37|
3DCNN Luo KAConv2D + KAN-Head| 	98.28	|99.41	|95.26	|89.08	|96.04	|93.99	|88.80|

### Spectral-spatial features (window size 7x7)

|Model|PaviaU*|PaviaC|Salinas|Indian Pines*|Houston13|Houston18|KSC|
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
3DCNN He|	98.05 |	97.39	| 93.03 |	84.45	| 90.40 | 	91.98|	88.87|
3DCNN He + KAN-Head|	98.54 |	99.34 |	96.70	|85.26	|92.93	|93.39	|89.65|

### Spectral-spatial features (window size 7x7)

|Model|PaviaU*|PaviaC|Salinas|Indian Pines*|Houston13|Houston18|KSC|
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
NM3DCNN |	99.33	|99.57 |	96.78 |	90.20 |	94.41 |	95.53 |	86.61 |
NM3DCNN + KAN-Head |	99.42 |	99.69 |	97.11 |	91.61 |96.92|	95.63 |	92.01|

### Spectral-spatial features (window size 13x13), 30 principal components

|Model|PaviaU*|PaviaC|Salinas|Indian Pines*|Houston13|Houston18|KSC|
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
SSFTT |	99.86 |	99.88	| 99.85 |	98.78 |	98.57 |	96.22 |	95.45 |
SSFTT KAConv2D + KAN-Head | 	99.92 |	99.92 |	99.93 |	99.13| 98.92	|96.55| 	97.34|


# Preprint

https://arxiv.org/abs/2407.05278
