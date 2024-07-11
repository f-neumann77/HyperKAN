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

*The training sample size for PaviaU and Indian Pines consisted of 20% samples per class, while for the other datasets 
it was 10% per class.

### Baseline 

|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
|MLP|	92.54     |	97.95     |	91.23   |	80.66	    | 95.06	    | 86.62     |	81.84     |
|KAN| **93.40** |	**98.53**	|**92.28**|	**83.70**	| **95.40**	| **87.70**	| **88.65** |

### Spectral-features
[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/)
|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
|1DCNN            |	94.21	|98.20	|91.93	|85.82	|92.63	|91.32	|84.87|
|1DCNN + KAN-Head |	95.63 |	99.04	|94.95	|87.17	|96.10	|93.00	|88.95|
|1DCNN full KAN   |**95.68**|	**99.07**|	**95.28**|	**88.90**	|**96.88**|	**93.63**	|**90.91**|

|Model | PaviaU*   | PaviaC      | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC |
|------|-----------|-------------|-----------|---------------|-----------|-----------|-----|
M1DCNN	          | 94.79	    | 98.81	      | 91.95	    | 84.22	        | 93.89	    |91.90	|85.67|
M1DCNN + KAN-Head | 	95.26    | 98.99       | 	92.83    | 	85.56        | 	96.75    |	**93.86** |	**91.12** |
M1DCNN full KAN	  | **95.45** | 	**99.06**	 | **93.51** | 	**86.08**	   | **96.83** |	93.34 |	90.04|

### Spectral-spatial features (window size 3x3)
[3DCNN by Luo et al](https://ieeexplore.ieee.org/document/8455251)
|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
3DCNN                    |	96.92 |	99.21 |	93.99	|81.90	|95.69	|92.45	|87.37|
3DCNN KAConv2D + KAN-Head| 	**98.28**	| **99.41**	| **95.26**	| **89.08**	| **96.04**	| **93.99**	| **88.80** |

### Spectral-spatial features (window size 7x7)
[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/)
|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
3DCNN He           |	98.05 |	97.39	| 93.03 |	84.45	| 90.40 | 	91.98|	88.87|
3DCNN He + KAN-Head|	**98.54** |	**99.34** |	**96.70**	| **85.26**	| **92.93**	| **93.39**	| **89.65** |

### Spectral-spatial features (window size 7x7)
|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
NM3DCNN            |	99.33	|99.57 |	96.78 |	90.20 |	94.41 |	95.53 |	86.61 |
NM3DCNN + KAN-Head |	**99.42** |	**99.69** |	**97.11** |	**91.61** | **96.92** |	**95.63** |	**92.01**|

### Spectral-spatial features (window size 13x13), 30 principal components
[SSFTT by Sun et al](https://ieeexplore.ieee.org/document/9684381)
|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
SSFTT                     |	99.86 |	99.88	| 99.85 |	98.78 |	98.57 |	96.22 |	95.45 |
SSFTT KAConv2D + KAN-Attention | 	**99.92** |	**99.92** |	**99.93** |	**99.13** | **98.92**	| **96.55** | **97.34** |


# Preprint

The results shown above may differ from the published preprint. The latter will be updated soon.

https://arxiv.org/abs/2407.05278
