# AI用オセロモジュール

## ファイル構成
root/
　├ datafile/
　│　└ kihuFixed.7z
　├ model/
　│　├ black.npz #先攻用モデル
　│　└ white.npz #後攻用モデル
　└ python/
　 　├ make_model.py #モデル作成用ファイル
　 　└ predict.py #予測用ファイル


## モデルの説明

X = [[[0,0,0,0,0,0,0,0],\
      [0,0,0,0,0,0,0,0],\
      [0,0,0,0,0,0,0,0],\
      [0,0,0,2,1,0,0,0],\
      [0,0,0,1,2,0,0,0],\
      [0,0,0,0,0,0,0,0],\
      [0,0,0,0,0,0,0,0],\
      [0,0,0,0,0,0,0,0]]] #盤面データ

Y = 20　#盤面での位置(左上1から始まって右下64までの連番)

- X,Yをセットで学習させたので、X型のデータを入れたらYの値が返却される。



## モデルの再作成
- パラメータや学習回数を変更させたい場合にはmake_model.pyの各値を変更すると新しい学習済モデルを作成できる

例
"""
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
                l1=L.Linear(64, 100), ← ここ(Linear)とか
                l2=L.Linear(100, 100),
                l3=L.Linear(100, 65),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))　← ここ(relu)とか
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
"""

"""
    optimizer = optimizers.SGD() ←ここ(SGD)
    optimizer.setup(model)
    
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (150, 'epoch'), out='result') ← ここ(epochの回数)とか
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png', trigger=(1,'epoch')))
    trainer.run()

"""

## 予測の仕方
- predict.pyで予測できる
- 盤面のデータと置ける場所(例:[[1,1],[1,2]])をそれぞれ配列で渡す