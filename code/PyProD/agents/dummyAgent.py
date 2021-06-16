from .Agent_Template import agent

class dummyAgent(agent):

    def __init__(self, bus1=None, bus2=None, line=None):
        # name (string) of the buses
        self.bus1 = bus1
        self.bus2 = bus2
        self.line = line
        self.model = self.build_model()
        self.inputs = None
        self.state = None
        self.obs = ['Vseq','Iseq']
        self.phases = 3
        self.svNum = None
        self.actNum = None
        self.trainable = False
        self.env = None

    # always do nothing 
    def act(self, obs):
        return 0

    # observe the case and update state container
    def observe(self, case):
        V = case.get_bus_V(self.bus1, 'Vseq', self.phases)
        I = case.get_line_I(self.line, 'Iseq', self.phases)

        self.state = V + I

    # reset the agent internal states
    def reset(self):
        self.state = None
