import argparse
import logging
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from network.policy_value_resnet import PolicyValueNetwork
from dataloader import HcpeDataLoader

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Train policy value network')
parser.add_argument('--train_data', type=str, default='train_data/train.hcpe', help='training data file')
parser.add_argument('--test_data', type=str, default='train_data/test.hcpe', help='test data file')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--checkpoint', default='checkpoints/checkpoint-{epoch:03}.pth', help='checkpoint file name')
parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
parser.add_argument('--eval_interval', type=int, default=100, help='evaluation interval')
parser.add_argument('--log', default=None, help='log file path')
args = parser.parse_args()

# ロギングの設定
logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('lr={}'.format(args.lr))

# デバイスの設定
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
logging.info(f"使用デバイス: {device}")

# モデルの初期化
model = PolicyValueNetwork()
model.to(device)

# オプティマイザの設定
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# 損失関数の設定
cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

# AMPのスケーラーを初期化
scaler = GradScaler()

# チェックポイントの読み込み
if args.resume:
    logging.info('Loading the checkpoint from {}'.format(args.resume))
    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint['epoch']
    t = checkpoint['t']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # 学習率を引数の値に変更
    optimizer.param_groups[0]['lr'] = args.lr
else:
    epoch = 0
    t = 0  # total steps

# 訓練データの読み込み
logging.info('Reading training data')
train_dataloader = HcpeDataLoader(args.train_data, args.batchsize, device, shuffle=True)
# テストデータの読み込み
logging.info('Reading test data')
test_dataloader = HcpeDataLoader(args.test_data, args.testbatchsize, device)

# 読み込んだデータ数をログに出力
logging.info('train position num = {}'.format(len(train_dataloader)))
logging.info('test position num = {}'.format(len(test_dataloader)))

# 方策の正解率を計算する関数
def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

# 価値の正解率を計算する関数
def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

# チェックポイントを保存する関数
def save_checkpoint():
    path = args.checkpoint.format(**{'epoch':epoch, 'step':t})
    logging.info('Saving the checkpoint to {}'.format(path))
    checkpoint = {
        'epoch': epoch,
        't': t,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


# 訓練ループ
for e in range(args.epoch):
    epoch += 1
    steps_interval = 0
    sum_loss_policy_interval = 0
    sum_loss_value_interval = 0
    steps_epoch = 0
    sum_loss_policy_epoch = 0
    sum_loss_value_epoch = 0
    for x, move_label, result in train_dataloader:
        model.train()

        # 順伝播と損失計算
        optimizer.zero_grad()
        with autocast():
            y1, y2 = model(x)
            loss_policy = cross_entropy_loss(y1, move_label)
            loss_value = bce_with_logits_loss(y2, result)
            loss = loss_policy + loss_value
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ステップ数と損失の更新
        t += 1
        steps_interval += 1
        sum_loss_policy_interval += loss_policy.item()
        sum_loss_value_interval += loss_value.item()

        # 評価間隔ごとの処理
        if t % args.eval_interval == 0:
            model.eval()
            x, move_label, result = test_dataloader.sample()
            with torch.no_grad():
                y1, y2 = model(x)
                test_loss_policy = cross_entropy_loss(y1, move_label).item()
                test_loss_value = bce_with_logits_loss(y2, result).item()
                test_accuracy_policy = accuracy(y1, move_label)
                test_accuracy_value = binary_accuracy(y2, result)

                logging.info(f'epoch = {epoch}, steps = {t}, train loss = {sum_loss_policy_interval / steps_interval:.07f}, {sum_loss_value_interval / steps_interval:.07f}, {(sum_loss_policy_interval + sum_loss_value_interval) / steps_interval:.07f}, test loss = {test_loss_policy:.07f}, {test_loss_value:.07f}, {test_loss_policy + test_loss_value:.07f}, test accuracy = {test_accuracy_policy:.07f}, {test_accuracy_value:.07f}')

            # エポックごとのステップ数カウンタと損失合計に加算
            steps_epoch += steps_interval
            sum_loss_policy_epoch += sum_loss_policy_interval
            sum_loss_value_epoch += sum_loss_value_interval

            # 評価間隔ごとのステップ数カウンタと損失合計をクリア
            steps_interval = 0
            sum_loss_policy_interval = 0
            sum_loss_value_interval = 0

    # エポックごとのステップ数カウンタと損失合計に加算
    steps_epoch += steps_interval
    sum_loss_policy_epoch += sum_loss_policy_interval
    sum_loss_value_epoch += sum_loss_value_interval

    # エポックの終わりにテストデータすべてを使用して評価する
    test_steps = 0
    sum_test_loss_policy = 0
    sum_test_loss_value = 0
    sum_test_accuracy_policy = 0
    sum_test_accuracy_value = 0
    model.eval()
    with torch.no_grad():
        for x, move_label, result in test_dataloader:
            y1, y2 = model(x)
            test_steps += 1
            sum_test_loss_policy += cross_entropy_loss(y1, move_label).item()
            sum_test_loss_value += bce_with_logits_loss(y2, result).item()
            sum_test_accuracy_policy += accuracy(y1, move_label)
            sum_test_accuracy_value += binary_accuracy(y2, result)

    logging.info(f'epoch = {epoch}, steps = {t}, train loss avr = {sum_loss_policy_epoch / steps_epoch:.07f}, {sum_loss_value_epoch / steps_epoch:.07f}, {(sum_loss_policy_epoch + sum_loss_value_epoch) / steps_epoch:.07f}, test loss = {sum_test_loss_policy / test_steps:.07f}, {sum_test_loss_value / test_steps:.07f}, {(sum_test_loss_policy + sum_test_loss_value) / test_steps:.07f}, test accuracy = {sum_test_accuracy_policy / test_steps:.07f}, {sum_test_accuracy_value / test_steps:.07f}')

    # チェックポイント保存
    if args.checkpoint:
        save_checkpoint()
