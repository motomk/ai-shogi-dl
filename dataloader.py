import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import torch
import torch.cuda as cuda

from cshogi import Board, HuffmanCodedPosAndEval
from features import FEATURES_NUM, make_input_features, make_move_label, make_result

# HcpeDataLoaderクラスの定義
class HcpeDataLoader:
    # 初期化メソッド
    def __init__(self, files, batch_size, device, shuffle=False):
        self.load(files)  # ファイルの読み込み
        self.batch_size = batch_size  # バッチサイズの設定
        self.device = device  # デバイスの設定
        self.shuffle = shuffle  # シャッフルの設定
        self.stream = cuda.Stream()  # GPU転送用のストリームを作成

        # テンソルの初期化
        self.torch_features = torch.empty((batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_move_label = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        # NumPy配列への変換
        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)

        self.i = 0  # インデックスの初期化
        self.executor = ThreadPoolExecutor(max_workers=4)  # スレッドプールエグゼキュータの設定

        self.board = Board()  # 将棋盤の初期化

    # ファイルの読み込みメソッド
    def load(self, files):
        data = []
        if type(files) not in [list, tuple]:
            files = [files]
        for path in files:
            if os.path.exists(path):
                logging.info(path)  # ログの記録
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logging.error('{} が見つかりません。スキップします。'.format(path))  # エラーログの記録
        if not data:
            raise FileNotFoundError('指定されたファイルが一つも見つかりませんでした。')
        self.data = np.concatenate(data)

    # ミニバッチの生成メソッド
    def mini_batch(self, hcpevec):
        self.features.fill(0)  # 特徴量の初期化
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])  # 局面の設定
            make_input_features(self.board, self.features[i])  # 入力特徴量の生成
            self.move_label[i] = make_move_label(hcpe['bestMove16'], self.board.turn)  # ムーブラベルの生成
            self.result[i] = make_result(hcpe['gameResult'], self.board.turn)  # 結果の生成

        with torch.cuda.stream(self.stream):  # 非同期転送を使用
            if self.device.type == 'cpu':
                return (self.torch_features.clone(),
                        self.torch_move_label.clone(),
                        self.torch_result.clone(),
                        )
            else:
                return (self.torch_features.to(self.device, non_blocking=True),
                        self.torch_move_label.to(self.device, non_blocking=True),
                        self.torch_result.to(self.device, non_blocking=True),
                        )

    # サンプルの取得メソッド
    def sample(self):
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    # プリフェッチメソッド
    def pre_fetch(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    # データの長さを返すメソッド
    def __len__(self):
        return len(self.data)

    # イテレータの初期化メソッド
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    # 次の要素を返すメソッド
    def __next__(self):
        if self.i > len(self.data):
            raise StopIteration()

        result = self.f.result()  # 結果の取得
        self.pre_fetch()

        return result
