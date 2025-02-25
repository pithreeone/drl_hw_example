# student_agent.py
import numpy as np
import pickle

# 載入訓練好的 Q-table
#with open("q_table.pkl", "rb") as f:
#    Q_table = pickle.load(f)

'''def get_action(obs):
    """ 使用 Q-table 選擇最佳行動 """
    return np.argmax(Q_table[obs])'''


import random
import gym

def get_action(obs):
    """ 隨機選擇一個動作 (0~5) """
    return random.choice([0, 1, 2, 3, 4, 5])
