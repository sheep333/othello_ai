#-- 作成したモデルを読み込んで実践 --#

#入ってくる値の変換機(gColを"A"などに変えても使える,rowは数値指定)
gCol = ('1','2','3','4','5','6','7','8')
gRow = ('1','2','3','4','5','6','7','8')

#-- 値を変換する関数を定義　--#
def conv_pos_to_num(position):
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


# 先攻だったらblack,後攻だったらwhiteのモデルを読み込む
# 指定しなければ通常の学習モデルが読み込まれる
start_with = "black"

model = Classifier(MLP())
if start_with == "black":
    serializers.load_npz("black.npz", model)
elif start_with == "white":
    serializers.load_npz("white.npz", model)
else:
    serializers.load_npz("model.npz", model)
  
#
def predict_best_pos(X1_,possible_num):
    best_num = 0 #置ける中でベストの位置
    best_predict = 0 #予測最高値
    for num in possible_num:
        y = F.softmax(model.predictor(np.array(X1_, dtype=np.float32)))
        if y.data[0][num] > best_predict:
            best_predict = y.data[0][num]
            best_num = num
    return conv_num_to_pos(best_num)

# test-data(こんなデータを入れてね！)
X1_ = [[[0,0,0,0,0,0,0,0],\
        [0,0,0,0,0,0,0,0],\
        [0,0,0,0,0,0,0,0],\
        [0,0,0,2,1,0,0,0],\
        [0,0,0,1,2,0,0,0],\
        [0,0,0,0,0,0,0,0],\
        [0,0,0,0,0,0,0,0],\
        [0,0,0,0,0,0,0,0]]]

possible_pos = [["4","3"],["3","4"],["6","5"],["5","6"]]

# 置ける場所の受け取り
possible_num = [conv_pos_to_num(position) for position in possible_pos]

# 予測関数
predict_best_pos(X1_,possible_num)