# ai-shogi-dl

## cshogi install
```
pip install git+https://github.com/TadaoYamaoka/cshogi
```

# 学習データのダウンロード
```N
wget http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor2020.7z
```

# csaからhcpeへの変換
```
python utils/csa_to_hcpe.py floodgate
```

## mcts_player.py 入力コマンド
```
setoption name debug value true
```
```
setoption name modelfile value checkpoints/checkpoint-001.pth
```
```
isready
```
```
position startpos
```
```
go byoyomi 1000
```

## 参考
こちらのコードを利用させていただきました。ありがとうございます。
https://github.com/TadaoYamaoka/cshogi
https://github.com/TadaoYamaoka/python-dlshogi2

