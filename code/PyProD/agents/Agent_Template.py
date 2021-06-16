# define an abstract class template of the Agent
class agent():
    def __init__(self, bus1=None, bus2=None):
        # name (string) of the buses
        self.bus1 = bus1
        self.bus2 = bus2
        self.successors = None
        self.model = self.build_model()
        self.inputs = None
        self.open = 0
        # observation should be a list of the following:
        # [Vseq, VLN, VLL, Iseq, Iph]
        self.obs = None
        self.phases = 3
        self.svNum = None
        self.actNum = None
        self.trainable = True
        self.rewardFcn = None

    # process the internal state and return model output
    # AbstractMethod
    def process_state(self):
        print(self.bus2)
        pass

    # convert a model ouput to the real trip command 0/1 (0 -> no trip, 1 -> trip)
    def act(self, action):
        # no wrapper by default
        trip = action
        return trip

    # adjust the agent's parameters according to the environment
    def configure(self, tier=None):
        pass


    # take a dssCase object and retrieve the observation it wants
    # return in desired format for the Agent's model
    def observe(self, case):
        pass


    # the environment should be set to train this agent, train the model if applicable
    def train(self):
        pass
    
    # 
    def build_model(self):
        pass
    
