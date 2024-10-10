import numpy as np

# 參數設定
alpha = 0.1  # 學習率
gamma = 0.9  # 折扣因子
beta = 0.1   # 鄰近狀態的影響因子
num_states = 10  # 狀態數量
num_actions = 2  # 動作數量（假設向左移動或向右移動）
q_table = np.zeros((num_states, num_actions))  # Q表格初始化

# 簡單的獎勵函數
def reward_function(state):
    if state == num_states - 1:
        return 10  # 終點狀態有較高的獎勵
    return -1  # 其他狀態的獎勵較低

# 鄰近狀態函數，假設鄰近狀態是相鄰的數字
def get_neighbors(state):
    neighbors = []
    if state > 0:
        neighbors.append(state - 1)
    if state < num_states - 1:
        neighbors.append(state + 1)
    return neighbors

# 簡單的 Q-learning 演算法，加入鄰近狀態影響
def update_q(state, action, reward, next_state):
    # 傳統 Q-learning 更新
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

    # 加入鄰近狀態的影響
    neighbors = get_neighbors(state)
    for neighbor in neighbors:
        q_table[state, action] += beta * (np.max(q_table[neighbor]) - q_table[state, action])

# 測試用的簡單模擬
for episode in range(100):
    state = np.random.randint(0, num_states)  # 隨機初始狀態
    done = False
    while not done:
        action = np.random.choice([0, 1])  # 隨機選擇動作：0向左，1向右
        next_state = max(0, min(num_states - 1, state + (1 if action == 1 else -1)))  # 移動到新狀態
        reward = reward_function(next_state)  # 根據新狀態取得獎勵
        update_q(state, action, reward, next_state)  # 更新 Q 值
        state = next_state
        if state == num_states - 1:  # 到達終點狀態
            done = True

# 印出最終的 Q 表格
print("最終的 Q 表格：")
print(q_table)
