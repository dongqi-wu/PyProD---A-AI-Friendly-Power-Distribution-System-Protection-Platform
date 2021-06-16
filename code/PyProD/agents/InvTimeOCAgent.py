from .Agent_Template import agent
import numpy as np


class OCAgent(agent):

    def __init__(self, bus1=None, bus2=None, line=None, Ipickup=None, TD_TMS=None, curve=None):
        # name (string) of the buses
        self.bus1 = bus1
        self.bus2 = bus2
        self.successors = None
        self.line = line
        self.model = self.build_model()
        self.tripped = False
        self.open = False
        self.state = None
        self.waveform = None
        self.actLog = None
        self.obs = ['Vseq','Iseq']
        self.phases = 3
        self.svNum = None
        self.actNum = None
        self.trainable = False
        self.env = None

        # OC parameters
        self.time = None
        self.triggerTime = None
        self.tripTime = None
        self.delay = None
        self.curve = curve
        self.td = TD_TMS
        self.Ip = Ipickup

    # return the time delay of this relay for a fault current
    def time_delay(self, Ifault):
        # ratio between actual and pickup current
        M = Ifault / self.Ip

        # IEC curve formulas (IEC 60255)
        if 'IEC' in self.curve:
            if self.curve == 'IEC-SIT': # standard inverse time
                k = 0.14
                alpha = 0.02
            elif self.curve == 'IEC-VIT': # very inverse
                k = 13.5
                alpha = 0.02
            elif self.curve == 'IEC-EIT': # extremely inverse
                k = 80
                alpha = 2
            elif self.curve == 'IEC-LTSI': # long time standard inverse
                k = 120
                alpha = 1
            else:
                raise ValueError(f'Please specify a valid Curve Type for Overcurrent relay at bus {self.bus1}!')

            return self.td * k / (M ** alpha - 1)

        # IEEE curve formulas (IEEE C37.112-1996)
        elif 'IEEE' in self.curve:
            if self.curve == 'IEEE-MIT': # moderately inverse
                A = 0.0515
                B = 0.114
                p = 0.02
            elif self.curve == 'IEEE-VIT': # very inverse
                A = 19.61
                B = 0.491
                p = 2
            elif self.curve == 'IEEE-EIT': # extremely inverse
                A = 28.2
                B = 0.1217
                p = 2
            else:
                raise ValueError(f'Please specify a valid Curve Type for Overcurrent relay at bus {self.bus1}!')

            return self.td * A / (M**p - 1) + B

        
    # Trip if I1 > Ipickup longer than the delay 
    def process_state(self):
        trip = 0
        if self.state > self.Ip:
            # if the relay is not triggered, trigger the relay
            if self.triggerTime == None:
                self.triggerTime = self.time
                self.delay = self.time_delay(self.state)
            # if the relay have been triggered for long enough, trip
            elif self.time - self.triggerTime >= self.delay:
                trip = 1
                self.tripped = True
                self.tripTime = self.time
        else:
            # if the current fells below Ip, stop the counting
            self.triggerTime = None
                    
        return trip

    # observe the case and update state container
    def observe(self, case):
        I = case.get_line_I(self.line, 'Iseq', self.phases)
        self.state = I[1] * np.random.uniform(0.985, 1.015) # positive seq current magnitude
        self.time = case.sol.Seconds
        self.waveform['I'].append(self.state)


    # reset the agent's internal states
    def reset(self):
        self.tripped = False
        self.open = False
        self.waveform = {'I':[]}
        self.state = None
        self.triggerTime = None
        self.delay = None


