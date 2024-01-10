import gym
import numpy as np
import random as rd
import os



def train():
      for i_episodes in range(20000):
          print("EPOCH：",i_episodes)
          # 重置游戏环境
          s = env.reset()
          i = 0
          # 学习 Q-Table
          while i < 2000:#步数
              i += 1
              if(s==(0, {'prob': 1})):
                  s=0
              # 使用带探索（ε-greedy）的策略选择动作
              a = epsilon_greedy(Q, s, i_episodes)
              # 执行动作，并得到新的环境状态、奖励等
              observation, reward, done, truncated, info= env.step(a)
              # 更新 Q-Table
          
              
              Q[s, a] = (1-learningRate) * Q[s, a] + learningRate * (reward + discountFactor * np.max(Q[observation, :]))
              s = observation
              if done:
                  break
              
def epsilon_greedy(q_table, s, num_episodes):
      rand_num = rd.randint(0, 20000)
      if rand_num > num_episodes:
          # 随机选择一个动作
          action = rd.randint(0, 3)
      else:
          # 选择一个最优的动作
          action = np.argmax(q_table[s, :])
      return action

def test(num):
    for i_episodes in range(num):
 # 重置游戏环境
        s = env.reset()
        i = 0
        total_reward = 0
        while i < 500:
            print(666)
            i += 1
            if(s==(0, {'prob': 1})):
                  s=0
    # 选择一个动作
            a = np.argmax(Q[s, :])
    # 执行动作，并得到新的环境状态、奖励等
            observation, reward, done, truncated, info= env.step(a)

            s = observation
            total_reward +=reward
            if done:
                rewardList.append(total_reward)
                break
        


current_directory = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_directory, 'Q.npy')
# 定义一个数组，用于保存每一回合得到的奖励
rewardList = []

mapmap0 = [
    "SHFFFFHF",
    "FFFHFHFF",
    "FHHHFFFF",
    "FHHHFFHF",
    "FFFFHHFF",
    "HFFHHHFF",
    "FHHHFFFH",
    "FFFHHFFG"
]
mapmap1 = [
    "SHFFFFHF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFHHFFG"
]

# 注册游戏环境

#-----------------------------------------------------------------训练保存
# env = gym.make('FrozenLake8x8-v1',is_slippery=False,desc=mapmap1)



# # 定义Q值表，初始值设为0
# Q = np.zeros([env.observation_space.n, env.action_space.n])
# # 设置学习参数
# learningRate = 0.85
# discountFactor = 0.95



# train()
# np.save(file_path, Q)
# test(100)  
# print("Final Q-Table Values：")
# print(Q)
# print("Success rate: " + str(sum(rewardList) / len(rewardList)))
#-----------------------------------------------------------------加载测试

env = gym.make('FrozenLake8x8-v1', render_mode="human",is_slippery=False,desc=mapmap1)
Q = np.load(file_path)
print(222)
test(2) 