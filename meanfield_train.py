import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from random import shuffle

# 設定設備為 GPU（如果可用），否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Softmax 策略
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

# 獲取自旋格點的子狀態
def get_substate(b):
    s = torch.zeros(2).to(device)
    if b > 0:
        s[1] = 1
    else:
        s[0] = 1
    return s

# 計算格點的平均行動
def mean_action(grid, j):
    x, y = get_coords(grid, j)
    action_mean = torch.zeros(2).to(device)
    for i in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            if i == k == 0:
                continue
            x_, y_ = x + i, y + k
            x_ = x_ if x_ >= 0 else grid.shape[0] - 1
            y_ = y_ if y_ >= 0 else grid.shape[1] - 1
            x_ = x_ if x_ < grid.shape[0] else 0
            y_ = y_ if y_ < grid.shape[1] else 0
            cur_n = grid[x_, y_]
            s = get_substate(cur_n)
            action_mean += s
    action_mean /= action_mean.sum()
    return action_mean

# 初始化自旋格子
def init_grid(size=(10, 10)):
    grid = torch.randint(0, 2, size).to(device)
    return grid

# 生成參數的函數
def gen_params(N, size):
    ret = []
    for i in range(N):
        vec = torch.randn(size).to(device) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret

# 假設的 q 函數，用來估算 Q 值
def qfunc(state, params, layers=[(2, 10), (10, 2)], afn=torch.tanh):
    l1n = layers[0]
    l1s = np.prod(l1n)
    theta_1 = params[0:l1s].reshape(l1n)
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = params[l1s:l2s + l1s].reshape(l2n)
    bias = torch.ones((1, theta_1.shape[1])).to(device)
    l1 = state @ theta_1 + bias
    l1 = torch.nn.functional.elu(l1)
    l2 = afn(l1 @ theta_2)
    return l2.flatten()

# 主程式開始
size = (10, 10)
J = np.prod(size)  # 計算格子的總數量
hid_layer = 10
layers = [(2, hid_layer), (hid_layer, 2)]
params = gen_params(1, 2 * hid_layer + hid_layer * 2)  # 生成隨機參數
grid = init_grid(size=size)
grid_ = grid.clone()
grid__ = grid.clone()

# 顯示初始狀態並保存為圖片
plt.imshow(grid.cpu())
plt.title("Initial Spin Lattice")
plt.colorbar()
plt.savefig("initial_grid.png")
plt.show()
print("Initial grid sum:", grid.sum())

# 開始訓練過程
epochs = 75
lr = 0.0001
num_iter = 3  # A
losses = [[] for i in range(J)]  # B
replay_size = 50  # C
replay = deque(maxlen=replay_size)  # D
batch_size = 10  # E
gamma = 0.9  # F

for i in range(epochs): 
    act_means = torch.zeros((J, 2)).to(device)  # G
    q_next = torch.zeros(J).to(device)  # H
    for m in range(num_iter):  # I
        for j in range(J):  # J
            action_mean = mean_action(grid_, j).detach()
            act_means[j] = action_mean.clone()
            qvals = qfunc(action_mean.detach(), params[0], layers=layers)
            action = softmax_policy(qvals.detach(), temp=0.5)
            grid__[get_coords(grid_, j)] = action
            q_next[j] = torch.max(qvals).detach()
        grid_.data = grid__.data
    grid.data = grid_.data

    # 計算當前動作和獎勵
    actions = torch.stack([get_substate(a.item()) for a in grid.flatten()])
    rewards = torch.stack([get_reward_2d(actions[j], act_means[j]) for j in range(J)])
    exp = (actions, rewards, act_means, q_next)  # K
    replay.append(exp)
    shuffle(replay)

    # 使用批量回放進行更新
    if len(replay) > batch_size:  # L
        ids = np.random.randint(low=0, high=len(replay), size=batch_size)  # M
        exps = [replay[idx] for idx in ids]
        for j in range(J):
            jacts = torch.stack([ex[0][j] for ex in exps]).detach()
            jrewards = torch.stack([ex[1][j] for ex in exps]).detach()
            jmeans = torch.stack([ex[2][j] for ex in exps]).detach()
            vs = torch.stack([ex[3][j] for ex in exps]).detach()
            qvals = torch.stack([qfunc(jmeans[h].detach(), params[0], layers=layers) for h in range(batch_size)])
            target = qvals.clone().detach()
            target[:, torch.argmax(jacts, dim=1)] = jrewards + gamma * vs
            loss = torch.sum(torch.pow(qvals - target.detach(), 2))
            losses[j].append(loss.item())
            loss.backward()
            with torch.no_grad():
                params[0] = params[0] - lr * params[0].grad
            params[0].requires_grad = True

# 可視化損失值和最終格子狀態
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(10, 10)

# 損失值的平均變化
mean_losses = np.mean([np.array(l) for l in losses if len(l) > 0], axis=0)
ax[0].plot(mean_losses)
ax[0].set_title("Average Loss over Epochs")

# 顯示最終自旋格子狀態
ax[1].imshow(grid.cpu())
ax[1].set_title("Final Spin Lattice")
plt.savefig('training_results.png')  # 保存圖片
plt.show()