from .Agent_Template import agent
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

class DeepNNRelayAgent(agent):
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
        self.svNum = 20
        self.actNum = None
        self.trainable = True

        # relay
        self.triggered = False
        self.triggerTime = None
        self.tripTime = None
        self.delay = 0.1
       
        # Deep NN parameters
        self.windowLength = 10
        self.build_model()
        self.action_space = None
        self.observation_space = None
        self.epiNum = 1000

    
    # process state and get model output
    def process_state(self):
        
        scores = self.model.predict(np.array(self.state).reshape(1, self.svNum))
        action = np.argmax(scores)
        # real action if delayed tripping is needed
        trip = 0
        if self.triggered:
            if action == 0:
                self.triggered = False
            elif self.time - self.triggerTime >= self.delay:
                trip = -1
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
        

                
    # build the NN model
    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=self.svNum, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        
    # observe the case and update state container
    def observe(self, case):
        I = list(case.get_line_I(self.line, 'Iseq', self.phases))
        V = list(case.get_bus_V(self.bus1, 'Vpuseq', self.phases))            
        self.time = case.sol.Seconds

        # store sequence current observation
        # add noise (assuming 3% IT accuracy)
        self.waveform['I1'].append(I[1] * np.random.uniform(0.985, 1.015))
        self.waveform['V1'].append(V[1] * np.random.uniform(0.985, 1.015) * 100)

        # format into self.state
        I1 = np.zeros(self.windowLength)
        V1 = np.zeros(self.windowLength)
        
        # number of observations available
        stepNum = len(self.waveform['I1'])
        if stepNum < self.windowLength:
            # fill all with initial value
            I1[:] = self.waveform['I1'][0]
            V1[:] = self.waveform['V1'][0]

            # fill the latest will real value
            I1[-stepNum:] = self.waveform['I1'][-stepNum:]
            V1[-stepNum:] = self.waveform['V1'][-stepNum:]
        else:
            I1[:] = self.waveform['I1'][-self.windowLength:]
            V1[:] = self.waveform['V1'][-self.windowLength:]

        self.state = np.concatenate([I1, V1], axis=0)

    # reset the agent's internal states
    def reset(self):
        self.tripped = False
        self.open = False
        self.waveform = {'I1':[],'V1':[]}
        self.actLog = []
        self.state = None
        self.triggerTime = None
        self.triggered = False
        
        # deep NN parameters
        self.Xs = [] # datas
        self.ys = [] # labels -- 0: no fault, 1: fault, 2:backup

    # use ProtEnv to create enough data, then train using sklearn
    def train(self, env):

        steps = 0
        ep = 0
        # run simulations until enough samples are collected
        while ep < self.epiNum:

            env.reset()
            ep += 1
            print(f'================ Episode {ep} ================')
            done = 0
            while env.currStep < env.maxStep and not done:
        
                
                # collect observations
                self.observe(env.case)
                
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
                    y = [0, 1, 0]
                # backup trip
                elif fault_bus_idx in self.successors and time_past > 0 and not coord_flag:
                    y = [0, 0, 1]
                # no trip
                else:
                    y = [1, 0, 0]

                    
                self.Xs.append(self.state)
                self.ys.append(y)
                # solve timestep
                env.currStep += 1
                env.case.solve_case()

            # train Deep NN model to forecast faults
            self.Xs = np.array(self.Xs)
            self.ys = np.array(self.ys)
            self.model.fit(self.Xs, self.ys, epochs=1, batch_size=1, verbose = 0)
        
    # save the model
    def save(self):
        pass
