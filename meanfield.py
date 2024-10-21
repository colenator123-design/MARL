import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from random import shuffle

# 生成參數的函數
def gen_params(N, size):
    ret = []
    for i in range(N):
        vec = torch.randn(size) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret

# 初始化自旋格子
def init_grid(size=(10, 10)):
    grid = torch.randint(0, 2, size)
    return grid

# 獲取自旋格點的子狀態
def get_substate(b):
    s = torch.zeros(2)
    if b > 0:
        s[1] = 1
    else:
        s[0] = 1
    return s

# Softmax策略
def softmax_policy(qvals, temp=0.9):
    soft = torch.exp(qvals / temp) / torch.sum(torch.exp(qvals / temp))
    action = torch.multinomial(soft, 1)
    return action

# 獲取格點在二維格子中的座標
def get_coords(grid, j):
    x = int(np.floor(j / grid.shape[0]))
    y = int(j - x * grid.shape[0])
    return x, y

# 計算二維格子的獎勵
def get_reward_2d(action, action_mean):
    r = (action * (action_mean - action / 2)).sum() / action.sum()
    return torch.tanh(5 * r)

# 計算格點的平均行動
def mean_action(grid, j):
    x, y = get_coords(grid, j)
    action_mean = torch.zeros(2)
    for i in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            if i == k == 0:
                continue
            x_, y_ = x + i, y + k
            # 處理邊界條件（週期性邊界）
            x_ = x_ if x_ >= 0 else grid.shape[0] - 1
            y_ = y_ if y_ >= 0 else grid.shape[1] - 1
            x_ = x_ if x_ < grid.shape[0] else 0
            y_ = y_ if y_ < grid.shape[1] else 0
            cur_n = grid[x_, y_]
            s = get_substate(cur_n)
            action_mean += s
    action_mean /= action_mean.sum()
    return action_mean

# 設置參數
size = (10, 10)  # 格子的大小
J = np.prod(size)  # 總共的格點數量
hid_layer = 10  # 隱藏層的大小
layers = [(2, hid_layer), (hid_layer, 2)]  # 神經網絡層結構
params = gen_params(1, 2 * hid_layer + hid_layer * 2)  # 生成隨機參數

# 初始化自旋格子
grid = init_grid(size=size)
grid_ = grid.clone()
grid__ = grid.clone()

# 顯示初始狀態並保存為圖片
plt.imshow(grid)
plt.title('Initial Spin Lattice')
plt.colorbar()

# 保存圖片到本地
plt.savefig('initial_grid.png')  # 保存成圖片檔案
print("Initial grid image saved as 'initial_grid.png'")

# 打印格子的自旋總和
print("Initial grid sum:", grid.sum())
