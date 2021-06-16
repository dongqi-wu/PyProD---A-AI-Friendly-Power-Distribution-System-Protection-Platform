from .Agent_Template import agent
import numpy as np
from sklearn import svm

class SVMRelayAgent(agent):
    def __init__(self, bus1=None, bus2=None, line=None):
        self.bus1 = bus1
        self.bus2 = bus2
        self.successors = None
        self.line = line
        self.tripped = False
        self.open = False
        self.waveform = None
        self.actLog = None
        self.obs = ['Iseq']
        self.phases = 3
        self.svNum = None
        self.actNum = None
        self.trainable = True

        # relay parameters
        self.triggered = False
        self.triggerTime = None
        self.tripTime = None
        self.delay = 0.1

        # SVM parameters
        self.model = svm.SVC(kernel='linear',C=1,decision_function_shape='ovr')
        self.action_space = None
        self.observation_space = None
        self.epiNum = 1000
        self.Xs = [] # datas
        self.ys = [] # labels -- 0: no fault, 1: fault
    
    # process state and get model output
    def process_state(self):
        V = self.waveform['V1'][-1]
        I = self.waveform['I1'][-1]

        action = self.model.predict([[I, V]])

        # real action if delayed tripping is needed
        trip = 0
        if self.triggered:
            if action == 0:
                self.triggered = False
            elif self.time - self.triggerTime >= self.delay:
                trip = 1
                self.tripped = True
                self.tripTime = self.time
        else:
            if action == 1:
                trip = 1
                self.tripped = True
                self.tripTime = self.time
            elif action == 2:
                self.triggerTime = self.time
                self.triggered = True

        return trip
        
        
    # observe the case and update state container
    def observe(self, case):
        I = list(case.get_line_I(self.line, 'Iseq', self.phases))
        V = list(case.get_bus_V(self.bus1, 'Vpuseq', self.phases))            
        self.time = case.sol.Seconds

        # store sequence current observation
        # add noise (assuming 3% IT accuracy)
        self.waveform['I1'].append(I[1] * np.random.uniform(0.985, 1.015))
        self.waveform['V1'].append(V[1] * np.random.uniform(0.985, 1.015) * 100)


    # reset the agent's internal states
    def reset(self):
        self.tripped = False
        self.open = False
        self.waveform = {'I1':[],'V1':[]}
        self.actLog = []
        self.state = None
        self.triggerTime = None
        self.triggered = False


    # use ProtEnv to create enough data, then train using sklearn
    def train(self, env):

        ep = 0
        # run simulations until enough samples are collected
        while ep < self.epiNum:

            env.reset()
            ep += 1
            print(f'================ Episode {ep} ================')
            done = 0
            while env.currStep < env.maxStep and not done:
        
                
                # collect observations
                Is = list(env.case.get_line_I(self.line, 'Iseq', self.phases))
                Vs = list(env.case.get_bus_V(self.bus1, 'Vpuseq', self.phases))    

                I = Is[1] * np.random.uniform(0.985, 1.015)
                V = Vs[1] * np.random.uniform(0.985, 1.015) * 100

                # compute label
                # is the time after fault?
                time_past = env.currStep * env.ts - env.fault.T

                # is the fault with area?
                fault_bus_idx = env.case.busNames.index(env.fault.bus)

                # is it backup or first?
                coord_flag = True
                for a in env.activeAgents:
                    a_bus_idx = env.case.busNames.index(env.agents[a].bus1)
                    if not env.agents[a].tripped and fault_bus_idx in env.agents[a].successors and a_bus_idx in self.successors:
                        coord_flag = False

                # instant trip
                if fault_bus_idx in self.successors and time_past > 0 and coord_flag:
                    y = 1
                # backup trip
                elif fault_bus_idx in self.successors and time_past > 0 and not coord_flag:
                    y = 2
                # no trip
                else:
                    y = 0    

                self.Xs.append([I, V])
                self.ys.append(y)
                # solve timestep
                env.currStep += 1
                env.case.solve_case()
                
            # train SVM model to forecast faults
            # 5 episodes per batch
            if ep % 20 == 0:
                self.model.fit(self.Xs, self.ys)
                self.Xs = [] # datas
                self.ys = [] # labels -- 0: no fault, 1: fault, 2: backup

                
        
    # save the model
    def save(self):
        pass
