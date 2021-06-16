import win32com.client
import networkx as nx
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from .utils import *
from rl.core import Env

# load a DSS case into the dssCase object
# The dssCase object provide a easy interface with OpenDSS and
# utility functions
class dssCase():
    def __init__(self, case_path, ts):
        # initialize DSS interface objects
        self.dss_handle = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.txt = self.dss_handle.Text
        self.ckt = self.dss_handle.ActiveCircuit
        self.sol = self.ckt.Solution
        self.ActElmt = self.ckt.ActiveCktElement
        self.ActBus = self.ckt.ActiveBus
        
        # load the case passed through argument
        self.case_path = case_path
        self.load_case()
        self.solve_case()
        self.check_grounding()

        # examine the network info of the DSS case and create a graph
        self.get_network_info()
        self.create_graph()
        self.sort_edges()
        self.posDict = None
        self.clrDict = None
        self.sizeDict = np.ones(self.busNum) * 30

        # simulation information
        self.ts = ts

    # clean DSS memory    
    def reset_dss(self):
        self.dss_handle.ClearAll()

    # load a local case from the case file
    def load_case(self):
        self.txt.Command = f"compile [{self.case_path}]"

    # solve the current loaded case
    def solve_case(self):
        self.sol.Solve()

    # process case and get network informations
    def get_network_info(self):
        # list of bus names
        self.busNames = self.ckt.AllBusNames
        self.busNum = len(self.busNames)
        # list of phases for each bus
        self.busPhases = []
        for n in self.busNames:
            self.ckt.SetActiveBus(n)
            self.busPhases.append(self.ActBus.Nodes)


        # list of lines
        self.lineNames = self.ckt.Lines.AllNames
        self.lineNum = self.ckt.Lines.Count

        self.lineT = []
        for n in self.lineNames:
            full_name = 'line.' + n
            self.ckt.SetActiveElement(full_name)
            F = self.ActElmt.Properties('Bus1').val.split('.')[0]
            T = self.ActElmt.Properties('Bus2').val.split('.')[0]
            
            # take only the 3-phase bus name 
            self.lineT.append((self.busNames.index(F),self.busNames.index(T)))

        # add transformers as lines (for graph making purpose)
        self.xfmrName = self.ckt.Transformers.AllNames
        self.xfmrNum = self.ckt.Transformers.Count
        self.xfmrT = []
        for tr in self.xfmrName:
            full_name = 'Transformer.' + tr
            self.ckt.SetActiveElement(full_name)
            F = self.busNames.index(self.ActElmt.busNames[0].split('.')[0])
            T = self.busNames.index(self.ActElmt.busNames[1].split('.')[0])

            self.xfmrT.append((F,T))
        self.xfmrT = list(set(self.xfmrT))

    # check if ground current path exist across the case
    def check_grounding(self):
        self.groundPath = False
        # go through transformers, loads and capactiors
        xfmrs = self.ckt.Transformers
        xfmrs.First
        for i in range(xfmrs.Count):
            if not xfmrs.IsDelta:
                self.groundPath = True
                return
            xfmrs.Next
            
        # capacitors
        caps = self.ckt.Capacitors
        caps.First
        for i in range(caps.Count):
            if not caps.IsDelta:
                self.groundPath = True
                return
            caps.Next
            
        # loads
        loads = self.ckt.Loads
        loads.First
        for i in range(loads.Count):
            if not loads.IsDelta:
                self.groundPath = True
                return
            loads.Next       
        
        

    ## GRAPH-RELATED FUNCTIONS
    # create a graph for the network using networkx
    def create_graph(self):
        # create new un-directed graph first
        self.graph = nx.Graph()
        
        # add lines as edges of this graph
        for l in self.lineT:
            self.graph.add_edge(l[0], l[1])

        # add transformers as edges of this graph
        for t in self.xfmrT:
            self.graph.add_edge(t[0], t[1])

        # change to directed graph and remove edges according to radial structure
        self.graph = self.graph.to_directed()

        ## for every line between bus, remove the backward edge that is not the assumed positive direction
        # compute the distance from source
        dist_from_source = nx.single_source_shortest_path_length(self.graph, 0) 

        # lines
        for l in self.lineT:
            # if bus1 is closer to souce than bus2
            if dist_from_source[l[0]] < dist_from_source[l[1]]:
                # remove the edge (bus2 -> bus1)
                try:
                    self.graph.remove_edge(l[1], l[0])
                except:
                    print(f'Warning: line {self.busNames[l[1]]} -> {self.busNames[l[0]]} cannot be removed, please check graph!')

        # transformers
        for t in self.xfmrT:
            # if bus1 is closer to souce than bus2
            if dist_from_source[t[0]] < dist_from_source[t[1]]:
                # remove the edge (bus2 -> bus1)
                try:
                    self.graph.remove_edge(t[1], t[0])
                except:
                    print(f'Warning: line {self.busNames[t[1]]} -> {self.busNames[t[0]]} cannot be removed, please check graph!')
        

    # draw the network graph using matplotlib
    def draw_graph(self):
        plt.figure()
        
        nx.draw(self.graph, pos=self.posDict, node_color=self.clrDict,\
                with_labels=False, node_size=self.sizeDict)
        
        plt.show()
        

    # sort nodes using DFS
    def sort_edges(self):
        self.edge_order = list(nx.dfs_edges(self.graph, source=0))


    # read bus coordinates from external file
    def read_bus_coords(self, fp):
        fh = open(fp, "r")
        allLines = fh.readlines()
        # guess the delimiter as there is no common format
        sn = csv.Sniffer()
        delim = sn.sniff(allLines[0]).delimiter
        
        # each line contains [busName, x, y] separated by space (?)
        self.posDict = {}
        for line in allLines:
            name, x, y = line.split(delim)
            busID = self.busNames.index(name)
            self.posDict.update({busID:[getNum(x),getNum(y)]})
            
    # mark the nodes that have protection problem as red
    def mark_logged_buses(self, log):

        # all bus where a fault is not detected
        allBuses = [self.busNames.index(l.fault.bus) for l in log]

        # paint these buses in red
        self.clrDict = []
        self.sizeDict = []
        for n in self.graph.nodes:
            if n in allBuses:
                self.clrDict.append('red')
                self.sizeDict.append(75)
            else:
                self.clrDict.append('blue')
                self.sizeDict.append(20)

        
    # get line current measurement using line name
    def get_line_I(self, name, field, phase):
        full_name = 'line.' + name
        self.ckt.SetActiveElement(full_name)
        if phase == 3:
            if field == 'Iseq':
                res = self.ActElmt.SeqCurrents[0:3]
            elif field == 'Ipuseq':
                res = self.ActElmt.SeqCurrents[0:3]
            elif field == 'Iph':
                res = cart_to_pol(self.ActElmt.Currents[0:6])
            else:
                raise ValueError(f'Please use a valid field name for Line measurement')
        # return both phase currents if 2 conductors
        elif phase == 2:
            res = cart_to_pol(self.ActElmt.Currents[0:4])
            
        # return single phase current if only 1 conductor
        elif phase == 1:
            res = cart_to_pol(self.ActElmt.Currents[0:2])
            
        return res
    
        
    # get bus voltage measurement using busname
    # returns:
    # [mag, angle]
    # mag = [magA, magB, magC]
    # angle = [angleA, angleB, angleC]
    # or
    # Sequence
    # [V0, V1, V2]
    def get_bus_V(self, name, field, phase):
        self.ckt.SetActiveBus(name)
        baseV = self.ActBus.kVbase * 1000
        if phase == 3:       
            if field == 'Vseq':
                res = self.ActBus.SeqVoltages
            if field == 'Vpuseq':
                res = [i / baseV for i in self.ActBus.SeqVoltages]
            elif field == 'VLN':
                mag, angle = cart_to_pol(self.ActBus.Voltages)
                res = [mag, angle]
            elif field == 'VLL':
                mag, angle = cart_to_pol(self.ActBus.VLL)
                res = [mag, angle]
            else:
                raise ValueError(f'Please use a valid field name for Bus measurement')

        elif phase == 2:
            if field == 'VLN':
                mag, angle = cart_to_pol(self.ActBus.Voltages)
                res = [mag, angle]
            elif field == 'VLL':
                mag, angle = cart_to_pol(self.ActBus.VLL)
                res = [mag, angle]
            else:
                raise ValueError(f'Please use a valid field name for Line measurement') 
    
        elif phase == 1:
            if field == 'VLN':
                mag, angle = cart_to_pol(self.ActBus.Voltages)
                res = [mag, angle]
            elif field == 'VLL':
                mag, angle = cart_to_pol(self.ActBus.VLL)
                res = [mag, angle]
            else:
                raise ValueError(f'Please use a valid field name for Line measurement')   

        
        return res

    # edit a property, or a list of properties, of a DSS element
    def edit_elmt(self, name, fields, vals):
        cmd = f'Edit {name}'
        # if providing a list of properties, iterate through
        if isinstance(fields, list):
            for f, v in zip(fields, vals):
                cmd += f' {f}={v}'
        # if only one property, add and execute
        else:
            cmd += f'{fields}={vals}'

        self.txt.Command = cmd

    # trip an element in the netwrok
    def trip_elmt(self, elmt):
        self.txt.Command = f'open line.{elmt} term=1'

    
    # create a random fault in this case
    def random_fault(self):
        randFault = fault(self.busNames, self.busPhases, self.ts, self.groundPath)
        
        return randFault

# log class for storing episode
class log():
    def __init__(self):
        self.fault = None
        self.tripTimes = None
        self.times = None
        self.agents_waveforms = []
        self.agents_actions = []

    # plot the record for agent a
    def visualize(self, a):
        all_waves = self.agents_waveforms[a]
        plt.figure()
        # plot waves
        for k in all_waves.keys():
            plt.plot(self.times, all_waves[k], label=k)
        # plot fault time and trip time
        plt.axvline(x = self.fault.T, color = 'r')
        if self.tripTimes[a] > 0:
            plt.axvline(x = self.tripTimes[a], color = 'b')
        plt.title(f'Waveform for Agent {a} for fault at {self.fault.bus}')
        plt.legend(all_waves.keys())
        plt.show()

    # plot the record for all agents
    def visualize_all(self):
        if len(self.agents_waveforms) == 1:
            self.visualize(0)
            return
        agentNum = len(self.tripTimes)
        fig, axs = plt.subplots(agentNum)
        fig.suptitle('Waveform of All Agents')
        for a in range(agentNum):
            # waveforms for this agent
            for k in self.agents_waveforms[a].keys():
                axs[a].plot(self.times, self.agents_waveforms[a][k], label=k)
            # fault time and trip time
            axs[a].axvline(x = self.fault.T, color = 'r')
            if self.tripTimes[a] > 0:
                axs[a].axvline(x = self.tripTimes[a], color = 'b')
            fig.legend(self.agents_waveforms[a].keys())
        plt.show()
        
# fault class for DSS
class fault():
    def __init__(self, buses, phases, ts, GNDFlag):
        self.GNDFlag = GNDFlag
        self.bus = self.rand_bus(buses[2:], phases[2:])
        self.phases = self.rand_phase(buses[2:], phases[2:])
        self.R = self.rand_resistance()
        self.T = self.rand_time(ts)
        self.cmd = self.get_cmd_string()

    # location of fault        
    def rand_bus(self, buses, phases):
        # return a random bus in the system
        self.bus_idx = np.random.choice(range(len(buses)))
        if self.GNDFlag:
            # 1 or 3-phase buses if GND path exists
            while not (len(phases[self.bus_idx])==1 or len(phases[self.bus_idx])==3):
                self.bus_idx = np.random.choice(range(len(buses)))
        else:
            # only 3-phase buses if GND path does not exist
            while not len(phases[self.bus_idx])==3:
                self.bus_idx = np.random.choice(range(len(buses)))            
        
        return buses[self.bus_idx]

    # return a fault type
    def rand_phase(self, buses, phases):
        p = phases[self.bus_idx]

        # if 1p line, only SLG  possible 
        if len(p) == 1:
            self.type = '1'
            return str(p[0])

        # if 2p line, SLG, LL or LLG
        if len(p) == 2:
            if self.GNDFlag:
                self.type = np.random.choice(['1','2'])
            else:
                self.type = '2'
                
            if self.type == '1':
                return np.random.choice(p)
            elif self.type == '2' or self.type == '2g':
                return np.random.choice(p, 2, replace=False)
        
        # if 3p line, can have all kinds of fault
        elif len(p) == 3:
            if self.GNDFlag:
                self.type = np.random.choice(['1','2','3'])
            else:
                self.type = np.random.choice(['2','3'])
                
            if self.type == '1':
                return np.random.choice(['1','2','3'])
            elif self.type == '2' or self.type == '2g':
                return np.random.choice(['1','2','3'], 2, replace=False)
            else:
                return ['1','2','3']

        
    def rand_resistance(self):
        # corresponding to low, med, high res fault
        fault_r_range = [[0.002,0.01],[0.01, 0.1],[0.1,1],[1,15]]
        fault_r = fault_r_range[np.random.choice([0,1,2,3])]
        #fault_r = fault_r_range[0]
        R = np.random.uniform(fault_r[0],fault_r[1])
        R = round(R, 4)
        return R

    def rand_time(self, ts):
        return round(((np.floor(np.random.uniform(15, 30))+0.1) * ts), 4)
        #return round((4.1 * ts), 4)

    # generate DSS command string from randomized attributes
    def get_cmd_string(self):
        cmd = 'New Fault.F1 '
        # number of phases
        cmd += 'Phases=' + str(len(self.phases))
        # format the faulted lines to the input form
        if self.type == '1':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0]
        elif self.type == '2':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0] + '.0'
            cmd += ' Bus2=' + self.bus + '.' + self.phases[1] + '.0'
        elif self.type == '2g':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0] + '.' + self.phases[0]
            cmd += ' Bus2=' + self.bus + '.' + self.phases[1] + '.0'
        elif self.type == '3':
            cmd += ' Bus1=' + self.bus + '.1.2.3'
        # fault resistance
        cmd += ' R=' + str(self.R)
        # fault time
        cmd += ' ONtime=' + str(self.T)

        return cmd
      

# main class for the relay environment
class rlEnv(Env):
    def __init__(self, case_path, agents, params=None):
        # unpack configuration dic
        self.ts = params['time_step']
        self.maxStep = params['max_step']
        self.case = dssCase(case_path, self.ts)
        self.caseName = case_path.split('\\')[-1].split('.')[0]

        self.loadProfile = None
        self.pvProfile = None
        self.windProfile = None
        self.DEREnable = params['DEREnable']

        # load load and DER profiles if supplied
        if not params['load_profile'] == None:
            self.loadProfile = pd.read_csv(params['load_profile'], index_col='date_time', parse_dates=True)

        if not params['pv_profile'] == None:
            self.pvProfile = pd.read_csv(params['pv_profile'], index_col='date_time', parse_dates=True)

        if not params['wind_profile'] == None:
            self.windProfile = pd.read_csv(params['wind_profile'], index_col='date_time', parse_dates=True)
        
        # store all agents that need to be trained
        self.agents = agents
        self.agentNum = len(self.agents)
        self.calc_agent_successors()
        self.trainingAgent = None
        self.activeAgents = None

        # sort and configure agents by location
        self.sort_agents()
        for a in range(self.agentNum):
            self.agents[a].configure(self.agentTier[a])

        # required fields for gym
        self.svNum = None
        self.actNum = None
        self.acton_space = None
        self.observation_space = None

        # environment state and containers
        self.logs = []
        

    # calculate the order of training
    # output a list of intergers of agents to be sorted
    # and    a list of intergers of the tiers of agents
    def sort_agents(self):
        # indicies of two terminals of the branch for each agent
        self.agentsBusIndex = [(self.case.busNames.index(a.bus1), self.case.busNames.index(a.bus2)) for a in self.agents]
    
        # find the index of line for each agent
        self.agentsLineIndex = []
        for a in self.agents:
            bus1_ind = self.case.busNames.index(a.bus1)
            bus2_ind = self.case.busNames.index(a.bus2)
            for l in self.case.lineT:
                if l[0] == bus1_ind and l[1] == bus2_ind:
                    self.agentsLineIndex.append(self.case.lineT.index(l))

        # find the position of each agent in the edge order from DFS
        # agent with the larger index is trained first
        agentPos = np.zeros(self.agentNum)
        for i in range(self.agentNum):
            agentPos[i] = self.case.edge_order.index(self.agentsBusIndex[i])
        sortedPos = -np.sort(-agentPos)

        self.train_order = [agentPos.tolist().index(i) for i in sortedPos]

        # compute the tier of agents (agents with the same tier can be trained in any order
        self.agentTier = np.zeros(self.agentNum, dtype='int')
        currTier = 0

        # first agent is tier 0
        self.agentTier[0] = currTier
        for a in self.train_order:
            # if agent a-1 is not a successor of a, they be long to the same tier
            if self.agentsBusIndex[a-1][0] in self.agents[a].successors:
                currTier += 1
            self.agentTier[a] = currTier
        

    # parse the network and get the successors of each agent
    def calc_agent_successors(self):
        for a in self.agents:
            a_bus_idx = self.case.busNames.index(a.bus1)
            a.successors = list(nx.nodes(nx.dfs_tree(self.case.graph, a_bus_idx)))
            a.successors.remove(a_bus_idx)

    # get measurements needed for an agent, specified in the fields of the agent class
    def take_sample(self, idx):
        all_sample = {}
        for i in self.agents[idx].obs:
            if i in ['Vseq', 'VLN', 'VLL']:
                ob = self.case.get_bus_V(self.agents[idx].bus1, i, self.agents[idx].phases)
            elif i in ['Iseq', 'Iph']:
                lineName = self.case.lineNames[self.agentsLineIndex[idx]]
                ob = self.case.get_line_I(lineName, i, self.agents[idx].phases)
            else:
                raise ValueError(f'Observation type not supported for agent{i}!')
                
            all_sample[i] = ob
            
        return all_sample


    # apply a randomly sampled profile to the system
    def apply_profile(self):
        # select an hour form the profiles
        if self.DEREnable:
            common_index = self.loadProfile.index.intersection(self.pvProfile.index).intersection(self.windProfile.index)
        else:
            common_index = self.loadProfile.index
        hour_num = np.random.choice(range(common_index.size))
        hour = common_index[hour_num]

        # apply load changes
        loadC = self.loadProfile.at[hour, 'value']
        if loadC < 0.3:
            loadC = 0.3

        loads = self.case.ckt.Loads
        loadNum = loads.Count
        # start from the first load
        loads.First
        for i in range(loadNum):
            loadP = loads.kW
            loadQ = loads.kvar
            loads.kW = loadP * loadC
            loads.kvar = loadQ * loadC
            loads.Next

        if self.DEREnable:
            # apply PV changes
            pvC = self.pvProfile.at[hour, 'value']

            # all pvs in the system
            pvs = self.case.ckt.PVSystems
            pvNum = pvs.Count
            
            # iterate through PVs
            # disable PVs if percentasge is too low
            pvs.First
            for i in range(pvNum):
                # disable PV if output is too low (for convergence)
                self.case.ckt.SetActiveElement(f'pvsystem.{pvs.Name}')
                if pvC < 0.1:
                    self.case.ckt.ActiveCktElement.Enabled = False
                else:
                    self.case.ckt.ActiveCktElement.Enabled = True
                pvVA = pvs.kVArated
                pvs.kVArated = pvVA * pvC
                pvs.Next


            # apply wind changes
            windC = self.windProfile.at[hour, 'value']
            # all gens in the system
            gens = self.case.ckt.Generators
            genNum = gens.Count

            # iterate through gens
            gens.First
            for i in range(genNum):
                self.case.ckt.SetActiveElement(f'generator.{gens.Name}')
                if windC < 0.1:
                    self.case.ckt.ActiveCktElement.Enabled = False
                else:
                    self.case.ckt.ActiveCktElement.Enabled = True                
                genP = gens.kW
                gens.kW = genP * windC
                gens.Next
            
        

    # this function is just for training, hense logic is tuned around the agent under training
    # next step giving the action of the relay under training
    def step(self, action):
        done = 0
        R = 0

        #print(self.case.sol.Seconds)

        # check for max simulation time
        if self.currStep == self.maxStep:
            done = 1

        # action of the current training agent
        train_trip = self.agents[self.trainingAgent].act(action)

        # compute reward for the training agent
        flags = self.assess_training_status()
        R = self.agents[self.trainingAgent].rewardFcn(flags, train_trip, self)
        
        if train_trip:
            done = 1
            self.agents[self.trainingAgent].open = True
            self.case.trip_elmt(self.agents[self.trainingAgent].line)

        # action of other active agents
        for a in self.activeAgents:
            act_temp = self.agents[a].process_state()
            a_trip = self.agents[a].act(act_temp)
            if a_trip:
                # if the tripping is successful
                #print(a, self.agents[a].triggerTime, self.agents[a].state)
                self.agents[a].open = True
                self.case.trip_elmt(self.agents[a].line)
 

        # solve this timestep
        self.currStep += 1
        self.case.solve_case()

        # get new observation for agents 
        for a in self.agents:
            a.observe(self.case)


        # DEBUG: print current and reward
        #print(self.agents[self.trainingAgent].state, train_trip, R, self.ts*self.currStep, self.agents[1].tripped)

        # return the observation of the agent under training
        ob_act = self.agents[self.trainingAgent].state
        
        
        return ob_act, R, done, {"Agent":self.trainingAgent}
                
        
    # clear DSS memory, reset the environment and start new episode
    def reset(self):
        # reset DSS
        self.case.reset_dss()
        self.case.load_case()
        self.case.sol.MaxIterations = 50
        self.case.sol.MaxControlIterations = 50

        # disable all fuses (TEMP)
        fuses = self.case.ckt.Fuses
        fuses.First
        for i in range(fuses.Count):
            fuses.RatedCurrent = 99999
            fuses.Next

        # change load and DER parameters
        self.apply_profile()

        # reset envrionment state
        self.tripTimes = np.zeros(self.agentNum)
        self.currStep = 1
        #print(self.case.sol.Seconds)

        # reset agent initial states
        for a in self.agents:
            a.reset()

        
        # add random fault
        self.fault = self.case.random_fault()
        self.case.txt.Command = self.fault.cmd

        # sample from load and DER profile


        # solve the initial power flow
        self.case.txt.Command = "set maxcontroliter=100"
        self.case.txt.Command = "set mode=snap"
        self.case.txt.Command = "Solve"

        assert self.case.sol.Converged, "Initial PF Failed!"
        
        # set dynamic mode
        self.case.txt.Command = "Solve mode=dynamics number=1 stepsize=" + str(self.ts)
        assert self.case.sol.Converged, "Dynamic PF Failed!"


        # get new observation for agents 
        for a in self.agents:
            a.observe(self.case)

        # return the observation of the agent under training
        if not self.trainingAgent == None:
            ob_act = self.agents[self.trainingAgent].state
        else:
            ob_act = None
            
        return ob_act
        
        
    # train all agents in this environment
    def train_agents(self):
        # activate not trainable agents
        self.activeAgents = []
        for a in range(self.agentNum):
            if not self.agents[a].trainable:
                self.activeAgents.append(a)

        # go through all agents in the list
        for a in self.train_order:
            # if a is not trainable, skip a
            if not self.agents[a].trainable:
                continue
            # else, train a
            else:
                print(f'================ Training {a} ================')
                # configure the env for this agent
                self.trainingAgent = a
                self.svNum = self.agents[a].svNum
                self.actNum = self.agents[a].actNum
                self.action_space = self.agents[a].action_space
                self.observation_space = self.agents[a].observation_space

                # train the agent
                self.agents[a].train(self)
                self.agents[a].save()
                # activate this trained agent
                if not a in self.activeAgents:
                    self.activeAgents.append(a)
                else:
                    raise ValueError(f'Training Error in agent{a}!')
            
 

    # evaluate all agents by running random eposides
    # return a log object defined in utils.py
    def evaluate(self, epiNum, verbose = True):

        # activate all agents
        self.activeAgents = range(self.agentNum)
        # go through episodes
        for ep in range(epiNum):
            
            self.reset()
            print(ep)
            if verbose:
                print(f'================ Episode {ep} ================')
                print(self.fault.cmd)           

            # loop steps
            done = 0
            while self.currStep < self.maxStep and not done:
                
                # get new observation for agents 
                for a in self.agents:
                    a.observe(self.case)
                    
                # collect the action of all agents
                for a in self.activeAgents:
                    act_temp = self.agents[a].process_state()
                    a_trip = self.agents[a].act(act_temp)

                    #flags = self.assess_training_status()
                    #R = self.agents[a].rewardFcn(flags, a_trip, self)
                    #print(R)

                    # record and trip the line if an agent has not tripped already
                    if a_trip and self.tripTimes[a] == 0:
                        self.tripTimes[a] = self.case.sol.Seconds
                        if verbose:
                            print(f'Relay {a} at bus {self.agents[a].bus1}, tripped at time {self.tripTimes[a]}!')
                        #print('line.' + self.agents[a].line)
                        done = 1
                        self.agents[a].open = True
                        self.case.trip_elmt(self.agents[a].line)

                # solve this timestep
                self.currStep += 1
                self.case.solve_case()

            agents_score = self.analyze_episode()
            # log only if incorrect operation happened
            if any([not i==1 for i in agents_score]):
                self.log_episode()
                if verbose:
                    print('Wrong operation! Episode have been logged')
                    


    # log the current episode 
    def log_episode(self):
        newLog = log()
        newLog.fault = self.fault
        newLog.tripTimes = self.tripTimes
        newLog.times = np.linspace(0, self.maxStep*self.ts, self.maxStep)
        for a in self.agents:
            newLog.agents_waveforms.append(a.waveform)
            newLog.agents_actions.append(a.actLog)

        self.logs.append(newLog)


    # analyze trip oprations associated with a fault
    # return a vector of length self.agentNum telling whether an operation is wrong
    # 1 - correct; 0 - incorrect
    def analyze_episode(self):
        # self.fault
        # self.tripTimes
        res = np.zeros(self.agentNum)
        # analyze active agents one by one
        for a in self.activeAgents:
            
            # is the fault within its designated area?
            fault_bus_idx = self.case.busNames.index(self.fault.bus)

            if fault_bus_idx in self.agents[a].successors:
                area_flag = True
            else:
                area_flag = False                
            
            # if this agent has operated in the past episode
            if self.tripTimes[a] > 0:
                
                # is the tripping after the fault?
                time_flag = self.tripTimes[a] > self.fault.T

                # is the fault within its designated area?
                # that is, does the faulted bus belong to the children of a relay
                fault_bus_idx = self.case.busNames.index(self.fault.bus)

                if fault_bus_idx in self.agents[a].successors:
                    area_flag = True
                else:
                    area_flag = False
                    
                # did it trip before another agent with higher priority?
                miscoord_flag = False
                for b in self.activeAgents:
                    # check other agents
                    if b == a:
                        pass
                    else:
                        # if it tripped before agent b
                        if self.tripTimes[a] < self.tripTimes[b]:
                            # check if b has higher priority, that is:
                            # 1) fault is a successor of b
                            # 2) b is a successor of a
                            b_bus_idx = self.case.busNames.index(self.agents[b].bus1)
                            if fault_bus_idx in self.agents[b].successors and b_bus_idx in self.agents[a].successors:
                                miscoord_flag = True

                # assert the correctness of agent a
                res[a] = time_flag and area_flag and not miscoord_flag
            # if this agent have not operated in the past episode
            else:
                # only if fault is outside the area or a neighbour has operated
                miscoord_flag = True
                for b in self.activeAgents:
                    # check other agents
                    if b == a:
                        pass
                    else:
                        # if b tripped and b is a successor of a, and b tripped after the fault
                        b_bus_idx = self.case.busNames.index(self.agents[b].bus1)
                        if fault_bus_idx in self.agents[b].successors and b_bus_idx in self.agents[a].successors and self.tripTimes[b] > self.fault.T:
                            miscoord_flag = False

                # assert the correctness of agent a
                res[a] = not area_flag or (area_flag and not miscoord_flag)
                
        return res

    # check circuit and fault status and return the condition of the training agent
    def assess_training_status(self):
        fault_bus_idx = self.case.busNames.index(self.fault.bus)

        # compute flags first

        # is the fault within designated area?
        if fault_bus_idx in self.agents[self.trainingAgent].successors:
            area_flag = True
        else:
            area_flag = False

        # is the time after fault?
        time_past = self.currStep * self.ts - self.fault.T
        if time_past > 0:
            time_flag = True
        else:
            time_flag = False

        # has the fault been cleared by a downstream neighbour or itself?
        clear_flag = False
        for a in self.activeAgents:
            a_bus_idx = self.case.busNames.index(self.agents[a].bus1)
            if self.agents[a].open and fault_bus_idx in self.agents[a].successors and a_bus_idx in self.agents[self.trainingAgent].successors:
                clear_flag = True
        # if the training agent has tripped
        if self.agents[self.trainingAgent].open and time_flag and area_flag:
            clear_flag = True

        # if not, is there a downstream neighbour that has not operated yet?
        # if every neighour has sent trip signal (need a backup tripping), this flag is True
        coord_flag = True
        for a in self.activeAgents:
            a_bus_idx = self.case.busNames.index(self.agents[a].bus1)
            if not self.agents[a].tripped and fault_bus_idx in self.agents[a].successors and a_bus_idx in self.agents[self.trainingAgent].successors:
                coord_flag = False

        flags = {'area': area_flag,
                 'time': time_flag,
                 'cleared': clear_flag,
                 'coord': coord_flag}

        return flags


    # close DSS and quit
    def close(self):
        self.case.reset_dss()
        
