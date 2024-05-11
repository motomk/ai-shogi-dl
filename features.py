"""
入力特徴量
(入力特徴量のチャネル数, 9, 9) 合計28チャネル
"""
import cshogi

# 移動方向を表す定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT,
    UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 入力特徴量の数
FEATURES_NUM = len(cshogi.PIECE_TYPES) * 2 + sum(cshogi.MAX_PIECES_IN_HAND) * 2

# 移動を表すラベルの数
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(cshogi.HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81


# 入力特徴量を作成
def make_input_features(board, features):
    # 入力特徴量を0に初期化
    features.fill(0)

    # 盤上の駒
    if board.turn == cshogi.BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 持ち駒
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, cshogi.MAX_PIECES_IN_HAND):
            features[i:i+num].fill(1)
            i += max_num


# 移動を表すラベルを作成
def make_move_label(move, color):
    to_sq = cshogi.move_to(move)
    from_sq = cshogi.move_from(move) if not cshogi.move_is_drop(move) else None

    # 後手の場合盤を回転
    if color == cshogi.WHITE:
        to_sq = 80 - to_sq
        if from_sq is not None:
            from_sq = 80 - from_sq

    if from_sq is not None:  # 駒の移動
        # 移動方向の計算
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y

        # 方向に基づいて移動方向を決定
        move_direction = determine_move_direction(dir_x, dir_y)

        # 成りの処理
        if cshogi.move_is_promotion(move):
            move_direction += 10
    else:  # 駒打ち
        move_direction = len(MOVE_DIRECTION) + cshogi.move_drop_hand_piece(move)

    return move_direction * 81 + to_sq


def determine_move_direction(dir_x, dir_y):
    if dir_y < 0:
        if dir_x == 0:
            return UP
        elif dir_y == -2 and dir_x == -1:
            return UP2_RIGHT
        elif dir_y == -2 and dir_x == 1:
            return UP2_LEFT
        elif dir_x < 0:
            return UP_RIGHT
        else:
            return UP_LEFT
    elif dir_y == 0:
        if dir_x < 0:
            return RIGHT
        else:
            return LEFT
    else:
        if dir_x == 0:
            return DOWN
        elif dir_x < 0:
            return DOWN_RIGHT
        else:
            return DOWN_LEFT


# 対局結果から報酬を作成
def make_result(game_result, color):
    if (color == cshogi.BLACK and game_result == cshogi.BLACK_WIN) or (color == cshogi.WHITE and game_result == cshogi.WHITE_WIN):
        return 1
    elif (color == cshogi.BLACK and game_result == cshogi.WHITE_WIN) or (color == cshogi.WHITE and game_result == cshogi.BLACK_WIN):
        return 0
    else:
        return 0.5
