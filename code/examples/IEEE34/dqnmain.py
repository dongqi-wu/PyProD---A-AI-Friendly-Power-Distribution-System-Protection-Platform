import sys
sys.path.append(r"C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code")
import PyProD.env as pe
from PyProD.agents.InvTimeOCAgent import OCAgent
from PyProD.agents.DQNAgent import DQNRelayAgent
from PyProD.rewards import step_reward
import numpy as np
import win32com.client

case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\IEEE34\ieee34Mod1_DER.dss'
#case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\OpenDSS\cases\123Bus\IEEE123Master.dss'
#case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\syn-austin-D_only-v03\P2U\base\opendss\p2uhs5_1247\p2uhs5_1247--p2udt7295\master.dss'


sim_params = {'time_step' : 1/60,
          'max_step' : 60,
          'DEREnable': True,
          'pv_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\pv_profile\ercot_houston_pv.csv',
          'wind_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\wind_profile\ercot_houston_wind.csv',
          'load_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\load_profile\ercot_houston_load.csv',
          }

## create agents
# RL agents
agent1_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\weights\IEEE34\800.h5f'
agent2_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\weights\IEEE34\802.h5f'
agent1 = DQNRelayAgent('800','802','l1', agent1_path, True)
agent2 = DQNRelayAgent('830','854','l15', agent2_path, True)
agents = [agent1, agent2]

agent1.rewardFcn = step_reward
agent2.rewardFcn = step_reward
## tell the environment about the agents
myEnv = pe.rlEnv(case_path, agents, sim_params)

#agent2.trainable = False
myEnv.train_agents()


myEnv.evaluate(2000, False)

# inspect logs
print(len(myEnv.logs))
fpNum = 0
fnNum = 0
mcNum = 0
for l in myEnv.logs:
    #l.visualize_all()
    #print(l.tripTimes)
    if any(t > 0 and t < l.fault.T for t in l.tripTimes):
        fpNum += 1
    if all(t <= 0 for t in l.tripTimes):
        fnNum += 1
    mcNum = len(myEnv.logs) - fpNum - fnNum

print(f'FP: {fpNum}, FN: {fnNum}, MC: {mcNum}, Total: {len(myEnv.logs)}')





## step through one episode
##done = 0
##myEnv.trainingAgent = 0 # first relay
##myEnv.activeAgents = [1]
##myEnv.reset()
##
##while not done:
##    act_temp = myEnv.agents[0].process_state()
##    act_trip = myEnv.agents[0].act(act_temp)
##    ob, R, done = myEnv.step(act_trip)


