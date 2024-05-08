import argparse
from cshogi import HuffmanCodedPosAndEval, Board, BLACK, move16
from cshogi import CSA
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--csa_dir', default='floodgate')
parser.add_argument('--hcpe_train', default='train_data/train.hcpe')
parser.add_argument('--hcpe_test', default='train_data/test.hcpe')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3500)
parser.add_argument('--test_ratio', type=float, default=0.1)
args = parser.parse_args()

# CSAファイルのリストを取得
csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

# ファイルリストをシャッフルして訓練データとテストデータに分割
file_list_train, file_list_test = train_test_split(csa_file_list, test_size=args.test_ratio)

# HuffmanCodedPosAndEvalの配列を初期化
hcpes = np.zeros(1024, HuffmanCodedPosAndEval)

# 訓練データとテストデータのファイルを開く
f_train = open(args.hcpe_train, 'wb')
f_test = open(args.hcpe_test, 'wb')

# 将棋盤の初期化
board = Board()
for file_list, f in zip([file_list_train, file_list_test], [f_train, f_test]):
    kif_num = 0
    position_num = 0
    for filepath in file_list:
        for kif in CSA.Parser.parse_file(filepath):
            # 棋譜のフィルタリング
            if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI') or \
               len(kif.moves) < args.filter_moves or \
               (args.filter_rating > 0 and min(kif.ratings) < args.filter_rating):
                continue

            # 開始局面を設定
            board.set_sfen(kif.sfen)
            p = 0
            try:
                for i, (move, score, comment) in enumerate(zip(kif.moves, kif.scores, kif.comments)):
                    # 合法手かどうかをチェック
                    if not board.is_legal(move):
                        raise Exception("Illegal move encountered")
                    # HuffmanCodedPosAndEvalオブジェクトを取得
                    hcpe = hcpes[p]
                    p += 1
                    # 局面情報をエンコード
                    board.to_hcp(hcpe['hcp'])
                    # 評価値をクリップして設定
                    eval_clipped = min(32767, max(score, -32767))
                    hcpe['eval'] = eval_clipped if board.turn == BLACK else -eval_clipped
                    # ベストムーブをエンコード
                    hcpe['bestMove16'] = move16(move)
                    # ゲーム結果を設定
                    hcpe['gameResult'] = kif.win
                    # 指し手を進める
                    board.push(move)
            except Exception as e:
                print(f'skip {filepath} due to {e}')
                continue

            # データをファイルに書き込む
            if p > 0:
                hcpes[:p].tofile(f)
                kif_num += 1
                position_num += p

    # 処理した棋譜数と局面数を出力
    print(f'kif_num: {kif_num}')
    print('position_num', position_num)
