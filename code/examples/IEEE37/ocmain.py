import sys
sys.path.append(r"C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code")
import ProtEnv.env as pe
from ProtEnv.agents.InvTimeOCAgent import OCAgent

import numpy as np
import win32com.client

case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\IEEE37\ieee37.dss'
#case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\OpenDSS\cases\123Bus\IEEE123Master.dss'
#case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\syn-austin-D_only-v03\P2U\base\opendss\p2uhs5_1247\p2uhs5_1247--p2udt7295\master.dss'


params = {'time_step' : 0.05,
          'max_step' : 300,
          'DEREnable' : False,
          'pv_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\pv_profile\caiso_la_pv.csv',
          'wind_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\wind_profile\caiso_la_wind.csv',
          'load_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\load_profile\caiso_la_load.csv',
          }

## create agents

# OC agents
agent1 = OCAgent('701','702','l1', 480, 0.1, 'IEEE-VIT')
agents = [agent1]

# tell the environment about the agents
myEnv = pe.rlEnv(case_path, agents, params)

# evaluate episodes
myEnv.evaluate(2000, False)

# inspect logs
print(len(myEnv.logs))
fpNum = 0
fnNum = 0
mcNum = 0
for l in myEnv.logs:
    l.visualize_all()
    #print(l.tripTimes)
    print(l.fault.cmd)
    if any(t > 0 and t < l.fault.T for t in l.tripTimes):
        fpNum += 1
    if all(t <= 0 for t in l.tripTimes):
        fnNum += 1
    mcNum = len(myEnv.logs) - fpNum - fnNum

print(f'FP: {fpNum}, FN: {fnNum}, MC: {mcNum}, Total: {len(myEnv.logs)}')

