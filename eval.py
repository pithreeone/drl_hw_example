# custom_taxi_env.py
import gym
import numpy as np
import time
import random
from IPython.display import clear_output

from xml.etree import ElementTree as ET
import importlib.util
import requests
import argparse

class DynamicTaxiEnv(gym.Wrapper):
    def __init__(self, grid_size=5, fuel_limit=50, randomize_passenger=True, randomize_destination=True, noise_level=0.1):
        self.grid_size = grid_size
        env = gym.make("Taxi-v3", render_mode="ansi")
        super().__init__(env)
        
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.randomize_passenger = randomize_passenger
        self.randomize_destination = randomize_destination
        self.noise_level = noise_level

        self.generate_random_map()

    def generate_random_map(self):
        """ 隨機產生站點與障礙物 """
        self.stations = random.sample([(x, y) for x in range(self.grid_size) for y in range(self.grid_size)], 4)
        self.obstacles = random.sample([(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.stations], int(self.grid_size**2 * 0.2))
        
        self.station_labels = ['R', 'G', 'Y', 'B']
        self.station_map = {self.station_labels[i]: self.stations[i] for i in range(4)}

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)  # Gym 0.26+ 的標準回傳值
        self.current_fuel = self.fuel_limit
        self.generate_random_map()

        # 設定乘客位置
        if self.randomize_passenger:
            self.passenger_loc = random.choice(self.stations)
        else:
            self.passenger_loc = self.stations[0]

        # 設定目的地
        if self.randomize_destination:
            possible_destinations = [s for s in self.stations if s != self.passenger_loc]
            self.destination = random.choice(possible_destinations)
        else:
            self.destination = self.stations[1]

        # 取得 Agent 目前的位置
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        # 轉換 `pass_idx` 與 `dest_idx` 為 (x, y) 座標
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination

        # 新的狀態表示方式 (可泛化到不同 `n × n` 地圖)
        state = (taxi_row, taxi_col, passenger_x, passenger_y, destination_x, destination_y)

        return state, info  # ✅ 確保回傳符合 Gym 0.26+

    def step(self, action):
        if np.random.rand() < self.noise_level:  # 讓 agent 有機率做錯誤的動作
            action = self.action_space.sample()

        # 取得 Agent 目前的位置
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(self.env.unwrapped.s)

        # 計算下一步位置
        next_row, next_col = taxi_row, taxi_col
        if action == 0:  # Move South
            next_row = min(self.grid_size - 1, taxi_row + 1)
        elif action == 1:  # Move North
            next_row = max(0, taxi_row - 1)
        elif action == 2:  # Move East
            next_col = min(self.grid_size - 1, taxi_col + 1)
        elif action == 3:  # Move West
            next_col = max(0, taxi_col - 1)

        # **檢查是否撞到障礙物**
        if (next_row, next_col) in self.obstacles:
            reward = -5  # 撞到障礙物
            self.current_fuel -= 1
            # 燃料檢查
            if self.current_fuel <= 0:
                return self.get_state(), reward -10, True, False, {}
            return self.get_state(), reward, False, False, {}

        # 燃料檢查
        if self.current_fuel <= 0:
            return self.get_state(), -10, True, False, {}

        # 執行動作
        self.current_fuel -= 1
        obs, reward, terminated, truncated, info = super().step(action)

        # 調整 reward（符合新規則）
        if reward == 20:  # 成功 `DROPOFF`
            reward = 50
        elif reward == -1:  # 每一步行動懲罰
            reward = -0.1
        elif reward == -10:  # 錯誤 `PICKUP` 或 `DROPOFF`
            reward = -10

        # 回傳自訂的 `state`
        return self.get_state(), reward, terminated, truncated, info

    def get_state(self):
        """ 取得當前 state (適用於不同 `n × n` 地圖) """
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(self.env.unwrapped.s)
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination
        return (taxi_row, taxi_col, passenger_x, passenger_y, destination_x, destination_y)

    def render_env(self, taxi_pos):
        """ 顯示環境狀態，標示 Agent 位置 """
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        for label, pos in self.station_map.items():
            grid[pos[0]][pos[1]] = label
        
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        grid[self.passenger_loc[0]][self.passenger_loc[1]] = 'P'
        grid[self.destination[0]][self.destination[1]] = 'D'
        grid[taxi_pos[0]][taxi_pos[1]] = '🚖'

        for row in grid:
            print(" ".join(row))
        print("\n")

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = DynamicTaxiEnv(**env_config)
    obs, _ = env.reset()  # obs 是 (taxi_row, taxi_col, passenger_x, passenger_y, destination_x, destination_y)
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        taxi_row, taxi_col, passenger_x, passenger_y, destination_x, destination_y = obs  # ✅ 正確解包 6 個值

        if render:
            print(f"step={step_count}")
            env.render_env((taxi_row, taxi_col))
            time.sleep(0.5)

        action = student_agent.get_action(obs)  # 確保 `get_action(obs)` 支援這個 state 格式
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

def parse_arguments():
    parser = argparse.ArgumentParser(description="HW1")

    parser.add_argument("--token", default="", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
   
    args = parse_arguments()

    # retrive submission meta info from the XML file
    xml_file_path = 'meta.xml'

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    sub_name = ""

    # Find the 'info' element and extract the 'name' value
    for book in root.findall('info'):
        sub_name =  book.find('name').text

    ### Start of evaluation section
    env_config = {
        "grid_size": 6,
        "fuel_limit": 10,
        "randomize_passenger": True,
        "randomize_destination": True,
        "noise_level": 0.1
    }

    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")
    ### End of evaluation section

    # push to leaderboard
    params = {
        'act': 'add',
        'name': sub_name,
        'score': str(agent_score),
        'token': args.token
    }
    url = 'http://localhost/drl_hw1/action.php'

    response = requests.get(url, params=params)
    if response.ok:
        print('Success:', response.text)
    else:
        print('Error:', response.status_code)