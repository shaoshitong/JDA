## Easy use see train.py ##

------------------------------

the log file for JDA in SEED is JDA.log

the acc for JDA in SEED is between 75% and 77%

Since the author's data preprocessing method is not public, I used Huan Wei's data preprocessing method, so the performance is very different from the 88% in the original paper.
use code:

### the way to optimize this model is that :###

Modify the learning rate, modify the batchsize, choose whether to regularize, choose the MSE function or Cross_entropy


### the way to run this model is that :###

```bash
git clone https://github.com/shaoshitong/JDA.git
cd ./JDA
git checkout origin/wangmingming -b wangmingming
python train.py
```

