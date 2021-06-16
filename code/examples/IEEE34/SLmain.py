import sys
sys.path.append(r"C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code")
import ProtEnv.env as pe
from ProtEnv.agents.InvTimeOCAgent import OCAgent
from ProtEnv.agents.DeepNNAgent import DeepNNRelayAgent
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
agent1 = DeepNNRelayAgent('800','802','l1')
agent2 = DeepNNRelayAgent('830','854','l15')
agents = [agent1, agent2]

## tell the environment about the agents
myEnv = pe.rlEnv(case_path, agents, sim_params)

#agent2.trainable = False
myEnv.train_agents()


myEnv.evaluate(500, True)

# inspect logs
print(len(myEnv.logs))
for l in myEnv.logs:
    print(l.fault.cmd)
    print(l.agents_actions)
    l.visualize_all()
    



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


