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

| Model                           | PaviaU* | PaviaC     | Salinas                        | Indian Pines*            | Houston13 | Houston18  | KSC       |
|---------------------------------|-------|------------|--------------------------------|--------------------------|--------|------------|-----------|
| MLP 64x64 (params: 22545)       | 	 90.59 | 	97.46     | 	90.10                         | 	68.51                   | 	76.82 | 	87.62     | 	72.66    |
| MLP 521x512 (params: 638 993)   | 	92.54 | 	97.95     | 	91.23                         | 	80.66	                  | 95.06	 | 86.62      | 	81.84    |
| KAN 64x64 (params: 182690)      | **94.92** | 	**98.62** | 	92.26| 	83.13| 	**95.45** | 	**88.49** | 	85.73    |
| KAN 512x512 (params: 3 755 042) | 93.40 | 	98.53	    | **92.28** | 	**83.70**	 | 95.40	 | 87.70	     | **88.65** |


### Spectral-features
[1DCNN by Hu et al](https://www.hindawi.com/journals/js/2015/258619/)

| Model                                   | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|-----------------------------------------|---------|--------|---------|---------------|-----------|-----------|-----|
| 1DCNN                (params: 378 353)  |	94.21	|98.20	|91.93	|85.82	|92.63	|91.32	|84.87|
| 1DCNN + KAN-Head     (params: 6 397 445)|	95.63 |	99.04	|94.95	|87.17	|96.10	|93.00	|88.95|
| 1DCNN full KAN (params: 6 472 163)      |**95.68**|	**99.07**|	**95.28**|	**88.90**	|**96.88**|	**93.63**	|**90.91**|

|Model | PaviaU*   | PaviaC      | Salinas   | Indian Pines* | Houston13 | Houston18 | KSC |
|------|-----------|-------------|-----------|---------------|-----------|-----------|-----|
M1DCNN	     (params: 346 653)     | 94.79	    | 98.81	      | 91.95	    | 84.22	        | 93.89	    |91.90	|85.67|
M1DCNN + KAN-Head (params: 5 997 102)| 	95.26    | 98.99       | 	92.83    | 	85.56        | 	96.75    |	**93.86** |	**91.12** |
M1DCNN full KAN	(params: 1 034 371)  | **95.45** | 	**99.06**	 | **93.51** | 	**86.08**	   | **96.83** |	93.34 |	90.04|

### Spectral-spatial features (window size 3x3)
[3DCNN by Luo et al](https://ieeexplore.ieee.org/document/8455251)

|Model | PaviaU* | PaviaC | Salinas | Indian Pines* | Houston13 | Houston18 | KSC |
|------|---------|--------|---------|---------------|-----------|-----------|-----|
3DCNN         (params: 54 817 499)            |	96.92       |	99.21   |	93.99 |81.90	  | 95.69    |92.45	   |87.37      |
3DCNN KAConv2D + KAN-Head (params: 550 614 188)| 	98.28	| 99.41	    | 95.26	  | 89.08	  | 96.04    | 93.99   | 88.80     |
3DCNN full KAN  (params: 48 479 367)          | **99.02**	| **99.57** |**96.97**| **91.33** |**96.77** |**94.16**| **90.72** | 

### Spectral-spatial features (window size 7x7)
[3DCNN by He et al](https://ieeexplore.ieee.org/document/8297014/)

|Model | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        |
|------|------------|------------|------------|---------------|------------|------------|------------|
3DCNN He        (params: 289 249)   | 	98.05     | 	97.39	    | 93.03      | 	84.45	       | 90.40      | 	91.98     | 	88.87     |
3DCNN He + KAN-Head (params: 85 300 978)| 	98.54     | 	99.34     | 	96.70	    | 85.26	        | 92.93	     | 93.39	     | 89.65      |
3DCNN He full KAN (params: 20 893 115)| 	**98.76** | 	**99.71** | 	**98.14** | 	**96.47**    | 	**95.73** | 	**95.82** | 	**93.66** |

### Spectral-spatial features (window size 7x7)
|Model | PaviaU*   | PaviaC     | Salinas    | Indian Pines* | Houston13   | Houston18  | KSC       |
|------|-----------|------------|------------|---------------|-------------|------------|-----------|
NM3DCNN        (params: 289 633)    | 	99.33	   | 99.57      | 	96.78     | 	90.20        | 	94.41      | 	95.53     | 	86.61    |
NM3DCNN + KAN-Head (params: 85 301 362)| 	99.42    | 	99.69     | 	97.11     | 	91.61        | 96.92       | 	95.63     | 	92.01    |
NM3DCNN full KAN (params: 10 520 631) | **99.52** | 	**99.75** | 	**98.01** | 	**95.39**    | 	**97.53**	 | **95.84**	 | **94.40** |

### Spectral-spatial features (window size 13x13), 30 principal components
[SSFTT by Sun et al](https://ieeexplore.ieee.org/document/9684381)

|Model | PaviaU*    | PaviaC     | Salinas | Indian Pines* | Houston13  | Houston18  | KSC        |
|------|------------|------------|--------|---------------|------------|------------|------------|
SSFTT          (params: 153 217)           | 	99.86     | 	99.88	    | 99.85  | 	98.78        | 	98.57     | 	96.22     | 	95.45     |
SSFTT KAConv2D + KAN-Attention (params: 15 408 209)| 	**99.92** | 	99.92     | 	99.93 | 	99.13        | 98.92      | 96.55      | 97.34      |
SSFTT full KAN (params: 338 107) | **99.92**  | 	**99.93** | **99.97** | 	**99.24**    | 	**99.46** | 	**97.12** | 	**98.76** |

# Preprint

The results shown above may differ from the published preprint. The latter will be updated soon.

https://arxiv.org/abs/2407.05278
