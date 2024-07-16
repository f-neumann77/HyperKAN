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

Parameters count is shown for Salinas dataset 

### Baseline 
first_hidden_layer X second_hidden_layer

| Model                             | PaviaU*                               | PaviaC     | Salinas                        | Indian Pines* | Houston13  | Houston18  | KSC       |
|-----------------------------------|---------------------------------------|------------|--------------------------------|-----------|------------|------------|-----------|
| MLP 8x8 (params: 1 937)           | 79.00                                 | 80.59      | 73.27                          | 42.98     | 53.35      | 72.45      | 48.83     |
| MLP 16x16 (params: 4 113)         | 	 79.15                               | 	96.52     | 	78.83                         | 	54.47    | 	74.23     | 	85.25     | 	59.92    |
| MLP 32x32 (params: 9 233)         | 	   87.79	                            | 97.34	     | 86.42	                         | 63.07	    | 76.00	     | 86.64	     | 67.43     |
| MLP 64x64 (params: 22 545)        | 	 90.59                               | 	97.46     | 	90.10                         | 68.51     | 	76.82     | 	87.62     | 	72.66    |
| MLP 128x128 (params: 61 457)      | 	  91.66 | 97.72      | 	90.07                         | 	73.63    | 	80.72     | 	87.00     | 	73.13    |    |
| MLP 256x256 (params: 188 433)     | 	   92.57	| 97.93	     | 90.55	                         | 76.80	    | 88.17	     | 87.18	     | 76.94     |
| MLP 512x512 (params: 638 993)     | 	92.54                                | 	97.95     | 	91.23                         | 	80.66    | 95.06	     | 86.62      | 	81.84    |
|  ||||||||
| KAN 8x8 (params: 18 386)          | 	    92.88| 	97.95     | 	90.60                         | 	76.63    | 	76.00     | 	**89.17** | 	71.57    |
| KAN 16x16 (params: 38 018)        | 	    93.95| 	98.51     | 	87.98                         | 	79.50    | 	93.07     | 	86.47     | 	76.85    |
| KAN 32x32 (params: 81 122)        | 	   94.88| 	98.69     | 	91.84                         | 	82.02    | 	**96.10** | 	87.90     | 	78.16    |
| KAN 64x64 (params: 182 690)       | **94.92**                             | 	**98.62** | 	92.26                         | 	83.13    | 	**95.45** | 	**88.49** | 	85.73    |
| KAN 128x128 (params: 447 266)     | 	  94.70| 	98.64     | 	92.88| 	84.35| 	95.40     | 	88.29     | 	84.30    |
| KAN 256x256 (params: 1 222 178)   | 	 94.65| 	98.77     | 	93.26     | 	83.87    | 	94.97     | 	87.32     | 	85.37     |
| KAN 512x512 (params: 3 755 042)   | 93.40                                 | 	98.53	    | **92.28**                      | 	**83.70**	 | 95.40	     | 87.70	     | **88.65** |


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


### Spatial-features (window size 3x3)

|Model | PaviaU*    | PaviaC     | Salinas    | Indian Pines* | Houston13  | Houston18  | KSC        |
|------|------------|------------|------------|---------------|------------|------------|------------|
M2DCNN | 	97.93     | 	99.03     | 	93.10	    | 86.56         | 	93.78     | 	93.19     | 	84.72     |
M2DCNN FULL KAN | 	**99.12** | 	**99.57** | 	**96.52** | 	**93.77**    | 	**95.97** | 	**94.26** | 	**86.19** |



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
