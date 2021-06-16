import sys
sys.path.append(r"C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code")
import ProtEnv.env as pe
from ProtEnv.agents.InvTimeOCAgent import OCAgent
from ProtEnv.agents.DeepNNAgent import DeepNNRelayAgent
import numpy as np
import win32com.client

case_path = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\syn-austin-D_only-v03\P1R\base\opendss\p1rhs4_1247\p1rhs4_1247--p1rdt6999\master.dss'
bus_coords = r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\case\syn-austin-D_only-v03\P1R\base\opendss\p1rhs4_1247\p1rhs4_1247--p1rdt6999\Buscoords.dss'

params = {'time_step' : 1/60,
          'max_step' : 60,
          'DEREnable' : False,
          'pv_profile' : None,
          'wind_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\wind_profile\ercot_houston_wind.csv',
          'load_profile' : r'C:\Users\Dongqi Wu\OneDrive\Work\PRRL\HICSS\code\data\load_profile\ercot_houston_load.csv',
          }


## create agents
# RL agents
agent1 = DeepNNRelayAgent('p1rdt6999-p1rhs4_1247x','p1rdt6999-p1rhs4_1247x_b1_1','l(r:p1rdt6999-p1rhs4_1247)')

agents = [agent1]

## tell the environment about the agents
myEnv = pe.rlEnv(case_path, agents, params)

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
    print(l.fault.cmd)
    if any(t > 0 and t < l.fault.T for t in l.tripTimes):
        fpNum += 1
    if all(t <= 0 for t in l.tripTimes):
        fnNum += 1
    mcNum = len(myEnv.logs) - fpNum - fnNum

print(f'FP: {fpNum}, FN: {fnNum}, MC: {mcNum}, Total: {len(myEnv.logs)}')
myEnv.case.mark_logged_buses(myEnv.logs)
myEnv.case.draw_graph()
