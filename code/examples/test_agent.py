import ProtEnv.env as pe
from ProtEnv.agents.InvTimeOCAgent import OCAgent
from ProtEnv.rewards import step_reward
import numpy as np
import win32com.client

case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\IEEE34\ieee34Mod1_DER.dss'
#case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\OpenDSS\cases\123Bus\IEEE123Master.dss'
#case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\syn-austin-D_only-v03\P2U\base\opendss\p2uhs5_1247\p2uhs5_1247--p2udt7295\master.dss'


params = {'time_step' : 1/60,
          'max_step' : 10,
          'DEREnable' : True,
          'pv_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\pv_profile\ercot_houston_pv.csv',
          'wind_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\wind_profile\ercot_houston_wind.csv',
          'load_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\load_profile\ercot_houston_load.csv',
          }

## create agents
agent1 = OCAgent('800','802','l1', 90, 0.2, 'IEEE-VIT')
agents = [agent1]

env = pe.rlEnv(case_path, agents, params)

env.reset()
print(env.fault.T / env.ts)
done = 0
while env.currStep < env.maxStep and not done:

    
    # collect observations
    agent1.observe(env.case)
    print(agent1.state)
    print(env.currStep)
    
    # solve timestep
    env.currStep += 1
    env.case.solve_case()
    input()
