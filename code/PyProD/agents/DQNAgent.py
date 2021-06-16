from .Agent_Template import agent
from gym import spaces
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class DQNRelayAgent(agent):
    def __init__(self, bus1=None, bus2=None, line=None, weightPath=None, loadWeights=False):
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
        self.loadWeights = loadWeights

        # customizable parameters
        self.verbose = 2
        self.lr = 0.0005
        self.min_delay = 0.05
        self.coord_step = 0.1 # sec
        self.trainingSteps = 30000
        self.windowLength = 12
        #self.counterDelays = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.3, 1.5, -1]
        #self.counterDelays = [0, 0.05, 0.1, -1]
        
        # learning parameters
        self.weightPath = weightPath
        self.svNum = 2 * self.windowLength + 1 # seq current in past 6 steps + triggered Flag
        self.state = None
        self.observation_space = spaces.Discrete(self.svNum)
        self.memory = SequentialMemory(limit=5000, window_length=1)
        self.policy = BoltzmannQPolicy()
        

        # relay parameters
        self.triggered = False
        self.triggerTime = None
        self.tripTime = None
        self.delay = None

    # configure and initialize the parameters based on environment
    def configure(self, tier):
        delays = [self.min_delay]
        for i in range(tier):
            delays.append(self.coord_step)
            #delays[0] = delays[0] + self.coord_step
        # add one more possible delay value for each tier            
        self.counterDelays = np.concatenate([delays, [-1]])
        self.actNum = len(self.counterDelays)
        self.action_space = spaces.Discrete(self.actNum)
        self.model = self.build_model()
        
        # load previous weights if provided
        if self.loadWeights:
            self.load()


    # build the DQN model for this agent
    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.svNum)))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.actNum))
        model.add(Activation('linear'))

        return model

    
    # process state and get model output
    def process_state(self):
        action = np.argmax(self.model.predict(self.state.reshape((1,1,self.svNum))))
        self.actLog.append(action)
        return action
        
    # compute real relay action of the agent using model action
    def act(self, action):
        actVal = self.counterDelays[action]
        trip = 0
        
        # if already open, no need to move further 
        if self.open:
            return trip
        
        # check if counter is triggered
        if self.triggered:
            # reset
            if actVal == -1:
                self.triggered = False
                self.triggerTime = None
            # check if need to trip (delay has passed)
            elif self.time - self.triggerTime >= self.delay:
                self.tripped = True
                self.triggered = False
                self.triggerTime = None
                self.tripTime = self.time
                trip = 1
        # if not triggered, check if need to be
        else:
            # if positive, set the countdown
            if actVal > 0:
                self.triggered = True
                self.triggerTime = self.time
                self.delay = actVal

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


        # concatenate  to form state vector
        if self.triggered:
            ct = self.delay - (self.time - self.triggerTime)
        else:
            ct = -1
        self.state = np.concatenate([I1, V1, [ct]], axis=0)
        
    

    # reset the agent's internal states
    def reset(self):
        self.tripped = False
        self.open = False
        self.waveform = {'I1':[],'V1':[]}
        self.actLog = []
        self.state = None
        self.triggerTime = None
        self.triggered = False
        self.delay = None


    # save the trained weight into local folder
    def save(self):
        print(f"Weights of agent {self.bus1} saved to {self.weightPath}")
        self.trainer.save_weights(self.weightPath, overwrite=True)

    # load
    def load(self):
        self.trainer = DQNAgent(self.model, \
                                nb_actions=self.actNum,
                                memory=self.memory, \
                                target_model_update=1e-3, \
                                policy=self.policy, \
                                nb_steps_warmup=1000, \
                                enable_double_dqn=True, \
                                enable_dueling_network=False)

        self.trainer.compile(Adam(learning_rate=self.lr), metrics=['mae'])
        self.trainer.load_weights(self.weightPath)

    # use keras-rl to train this agent
    def train(self, env):
        self.trainer = DQNAgent(self.model, \
                                nb_actions=self.actNum,
                                memory=self.memory, \
                                target_model_update=1e-3, \
                                policy=self.policy, \
                                nb_steps_warmup=1000, \
                                enable_double_dqn=True, \
                                enable_dueling_network=False)

        self.trainer.compile(Adam(learning_rate=self.lr), metrics=['mae'])
        self.hist = self.trainer.fit(env, nb_steps=self.trainingSteps, visualize=False, verbose=self.verbose)



        
    
