#-- 作成したモデルを読み込んで実践 --#
import sys,ast
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#-- 値を変換する関数を定義　--#
def conv_pos_to_num(position):
    print(position)
    col_num = gCol.index(position[0])+1
    row_num = int(position[1])
    pos_num = (row_num-1)*8 + col_num 
    return pos_num

def conv_num_to_pos(pos_num):
    pos = []
    row,col= divmod(pos_num,8)
    if col == 0:
      pos.append(gcol[7])
      pos.append(row)
    else:
      pos.append(gCol[col-1])    
      pos.append(str(row+1))  
    return pos


# 分類器作成
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
                l1=L.Linear(64, 100),
                l2=L.Linear(100, 100),
                l3=L.Linear(100, 65),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

#-- 必要な関数を定義　--#
pos = []
def conv_pos_to_num(position):
    try:
        pos_num = position[0] + (position[1]-1)*8
    except:
        print("error")
        pos_num = 0
        
    return pos_num

def conv_num_to_pos(pos_num):
    row,col= divmod(pos_num,8)
    if col == 0:
        pos.append(8)
        pos.append(row+1)
    else:
        pos.append(col)
        pos.append(row+1)
    return pos

# コマンドラインからの引数受け取り
argvs = sys.argv
argc = len(argvs)

# 先攻だったらblack,後攻だったらwhiteのモデルを読み込む
# 指定しなければ通常の学習モデルが読み込まれる
start_with = argvs[1]
# test-data(こんなデータを入れてね！)
X1_ = [[0,0,0,0,0,0,0,0],\
       [0,0,0,0,0,0,0,0],\
       [0,0,0,0,0,0,0,0],\
       [0,0,0,2,1,0,0,0],\
       [0,0,0,1,2,0,0,0],\
       [0,0,0,0,0,0,0,0],\
       [0,0,0,0,0,0,0,0],\
       [0,0,0,0,0,0,0,0]]

possible_pos = [[4,3],[3,4],[6,5],[5,6]]

X1_ = [ast.literal_eval(argvs[2])]
possible_pos = ast.literal_eval(argvs[3])

if (argc != 4):
    print('引数が不正です。')
    quit()

model = Classifier(MLP())
if start_with == "black":
    serializers.load_npz("../model/black.npz", model)
elif start_with == "white":
    serializers.load_npz("../model/white.npz", model)
else:
    serializers.load_npz("../model/model.npz", model)


def predict_best_pos(X1_,possible_num):
    best_num = 0 #置ける中でベストの位置
    best_predict = 0 #予測最高値
    for num in possible_num:
        y = F.softmax(model.predictor(np.array(X1_, dtype=np.float32)))
        if y.data[0][num] > best_predict:
            best_predict = y.data[0][num]
            best_num = num
    return conv_num_to_pos(best_num)

# 置ける場所の受け取り
possible_num = [conv_pos_to_num(position) for position in possible_pos]

# 予測関数
result = predict_best_pos(X1_,possible_num)

print(result)
