from scipy.io import loadmat
from data.dataloader import get_dataloader
from models.vanilla_models.vanilla_1DCNN import Model1DCNN

img = loadmat('./test_data/PaviaU.mat')['paviaU']
gt = loadmat('./test_data/PaviaU_gt.mat')['paviaU_gt']


train_dataloader = get_dataloader(img=...,
                                  gt=train_gt,
                                  model_name='1DCNN',
                                  hyperparams=...,
                                  shuffle=...)

val_dataloader = get_dataloader()

test_dataloader = get_dataloader(img=...,
                                 gt=test_gt,
                                 model_name='1DCNN',
                                 hyperparams=...,
                                 shuffle=...)

nn = Model1DCNN()

nn.fit(train_dataloader)

predictions = nn.predict(test_dataloader)



