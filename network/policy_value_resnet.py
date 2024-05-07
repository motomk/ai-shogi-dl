import torch
import torch.nn as nn
import torch.nn.functional as F

# 特徴量の数、移動プレーン数、移動ラベル数をインポート
from features import FEATURES_NUM, MOVE_PLANES_NUM, MOVE_LABELS_NUM

# バイアスを追加するクラス
class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))  # バイアスパラメータの初期化

    def forward(self, input):
        return input + self.bias  # 入力にバイアスを加える

# ResNetブロッククラス
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)  # 1つ目の畳み込み層
        self.bn1 = nn.BatchNorm2d(channels)  # 1つ目のバッチ正規化
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)  # 2つ目の畳み込み層
        self.bn2 = nn.BatchNorm2d(channels)  # 2つ目のバッチ正規化

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)  # 活性化関数ReLUを適用

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + x)  # 残差接続とReLUの適用

# ポリシーと価値のネットワーククラス
class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, channels=192, fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=FEATURES_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)  # 初期畳み込み層
        self.norm1 = nn.BatchNorm2d(channels)  # 初期バッチ正規化

        # ResNetブロックのシーケンス
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        # ポリシーヘッド
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False)  # ポリシー畳み込み層
        self.policy_bias = Bias(MOVE_LABELS_NUM)  # ポリシーバイアス

        # バリューヘッド
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False)  # バリュー畳み込み層
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)  # バリューバッチ正規化
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)  # バリュー全結合層1
        self.value_fc2 = nn.Linear(fcl, 1)  # バリュー全結合層2

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.norm1(x))  # 活性化関数ReLUの適用

        # ResNetブロックを通過
        x = self.blocks(x)

        # ポリシーヘッドの処理
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))  # バイアスの適用とフラット化

        # バリューヘッドの処理
        value = F.relu(self.value_norm1(self.value_conv1(x)))  # 活性化関数ReLUの適用
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))  # フラット化と活性化関数ReLUの適用
        value = self.value_fc2(value)  # 最終的なバリュー出力

        return policy, value
