from collections import deque
import copy
import torch


class EnvironmentDirect(object):
    def __init__(self, H, P, T, dGen, miniBatch, nodes, edges, downwardNetwork, upwardNetwork, DNNEdges, edgeStrategy
                 , device, threshold, initializtion, LS=0, LO=0, verbosity=0, demand_mean=5,
                 approach='OUL', salvageValue=0):
        '''
        H: Holding cost vector
        P: Stock-out cost vector
        T: Time horizon
        dGen: distribution generator (function)
        nodes: nodes of the network
        edges: edges of the network
        downwardNetwork: the supply network (shipment point of view)
        upwardNetwork: the supply network (order point of view)
        node: under study node
        LS: Shipment Lead time
        LO: Order Lead time
        '''
        self.H = H
        self.P = P
        self.T = T
        self.dGen = dGen
        self.num_env = miniBatch
        self.nodes = nodes
        self.edges = edges
        self.downwardNetwork = downwardNetwork
        self.upwardNetwork = upwardNetwork
        self.threshold = threshold
        self.DNNEdges = DNNEdges  # like: [(1,2),(3,4)]
        self.edgeStrategy = edgeStrategy
        self.initializtion = initializtion
        self.LS = LS
        self.LO = LO
        self.verbosity = verbosity
        self.device = device
        self.demand_mean = demand_mean
        self.approach = approach
        self.salvageValue = salvageValue
        self.reset()

    def reset(self):
        self.ILs = {i: {j: torch.zeros(self.num_env).to(device=self.device) + self.initializtion[i]
        [self.upwardNetwork[i].index(j)] for j in self.upwardNetwork[i]} for i in
                    self.upwardNetwork.keys()}  # Inventory Level base-stock initialization
        self.Ss = {i: {j: torch.zeros(self.num_env).to(device=self.device) for j in self.upwardNetwork[i]} for i in
                   self.upwardNetwork.keys()}
        self.AOs = {i: {j: deque([torch.zeros(self.num_env).to(device=self.device) for _ in
                                  range(self.LO[i][self.downwardNetwork[i].index(j)] + 1)],
                                 self.LO[i][self.downwardNetwork[i].index(j)] + 1) for j in self.downwardNetwork[i]} for
                    i in self.downwardNetwork.keys()}  # Arriving Order from downstream
        self.ASs = {i: {j: deque([torch.zeros(self.num_env).to(device=self.device) for _ in
                                  range(self.LS[i][self.upwardNetwork[i].index(j)] + 1)],
                                 self.LS[i][self.upwardNetwork[i].index(j)] + 1) for j in self.upwardNetwork[i]} for i
                    in self.upwardNetwork.keys()}  # Arriving Shipment from upstream
        self.OOs = {i: {j: torch.zeros(self.num_env).to(device=self.device) for j in self.upwardNetwork[i]} for i in
                    self.upwardNetwork.keys()}  # On-Order quantity
        self.OSs = {i: {j: torch.zeros(self.num_env).to(device=self.device) for j in self.downwardNetwork[i]} for i in
                    self.downwardNetwork.keys()}  # Out-bound Shipment to downstream
        self.Qs = {i: {j: torch.zeros(self.num_env).to(device=self.device) for j in self.upwardNetwork[i]} for i in
                   self.upwardNetwork.keys()}  # order quantity  (determined by base-stock or RL)
        self.t = 0  # step time
        if self.verbosity == 2:
            self.BLvls = {i: {j: [] for j in self.upwardNetwork[i]} for i in
                          self.upwardNetwork.keys()}  # Possible base-Stock levels
            self.ILPrinting = {i: {j: [] for j in self.upwardNetwork[i]} for i in self.upwardNetwork.keys()}
            self.IPPrinting = {i: {j: [] for j in self.upwardNetwork[i]} for i in self.upwardNetwork.keys()}
            self.QPrinting = {i: {j: [] for j in self.upwardNetwork[i]} for i in self.upwardNetwork.keys()}
        self.totalReward = []

    def step(self, teacher, history):

        done = False
        currentBackorderList = []
        for node in self.downwardNetwork.keys():
            if self.downwardNetwork[node] == [0]:
                self.AOs[node][0].append(self.dGen().to(device=self.device))
        WholeILs = []
        for node in reversed(sorted(self.upwardNetwork.keys())):
            for adjacentNode in self.upwardNetwork[node]:
                WholeILs.append(torch.unsqueeze(self.ILs[node][adjacentNode], 1))

        WholeILs = torch.cat(WholeILs, axis=1)
        if self.edgeStrategy == 'DNN':
            actions = self.TakeAction(WholeILs, teacher)
        for node in reversed(sorted(self.upwardNetwork.keys())):
            for adjacentNode in self.upwardNetwork[node]:
                self.Ss[node][adjacentNode] = self.ILs[node][adjacentNode] + self.OOs[node][adjacentNode] - sum \
                    ([self.AOs[node][i][0] for i in self.AOs[node].keys()])
                s = copy.copy(torch.unsqueeze(self.Ss[node][adjacentNode], 1))
                history[(adjacentNode, node)]['s'].append(s)
                if (adjacentNode, node) in self.DNNEdges and self.edgeStrategy == 'DNN':
                    a = actions[(adjacentNode, node)]

                else:
                    if self.approach == 'OUL':
                        a = self.BSAction(torch.zeros_like(s), self.threshold[(adjacentNode, node)])
                    else:
                        a = self.BSAction(s, self.threshold[(adjacentNode, node)])

                if self.approach == 'OUL':
                    self.Qs[node][adjacentNode] = (torch.squeeze(self.BSAction(s, a)))
                else:
                    self.Qs[node][adjacentNode] = (torch.squeeze(a))

                self.OOs[node][adjacentNode] = self.OOs[node][adjacentNode] + self.Qs[node][adjacentNode]
                if 0 not in self.upwardNetwork[node]:
                    self.AOs[adjacentNode][node].append(self.Qs[node][adjacentNode])
                history[(adjacentNode, node)]['a'].append(a[0])

        for node in self.upwardNetwork.keys():
            if self.upwardNetwork[node] == [0]:
                self.ASs[node][0].append(self.Qs[node][0])

        total_reward = torch.zeros(self.num_env).to(device=self.device)
        for node in sorted(self.upwardNetwork.keys()):
            for adjacentNode in self.upwardNetwork[node]:

                self.OOs[node][adjacentNode] = self.OOs[node][adjacentNode] - self.ASs[node][adjacentNode][0]

                currentInv = torch.max(torch.zeros(len(self.ILs[node][adjacentNode])).to(device=self.device),
                                       self.ILs[node][adjacentNode]) + self.ASs[node][adjacentNode][0]
                currentBackorders = torch.max(torch.zeros(len(self.ILs[node][adjacentNode])).to(device=self.device)
                                              , -self.ILs[node][adjacentNode])
                currentBackorderList.append(currentBackorders)
                self.ILs[node][adjacentNode] = self.ILs[node][adjacentNode] + self.ASs[node][adjacentNode][0]
                newDemand = sum([self.AOs[node][i][0] for i in self.AOs[node].keys()])

                self.ILs[node][adjacentNode] = self.ILs[node][adjacentNode] - newDemand

                reward = - 1 * (self.H[(adjacentNode, node)] * torch.max
                (torch.zeros(len(self.ILs[node][adjacentNode])).to(device=self.device)
                 , self.ILs[node][adjacentNode]).to(device=self.device) + self.P[(adjacentNode, node)] * torch.max
                                (torch.zeros(len(self.ILs[node][adjacentNode])).to(device=self.device)
                                 , - 1 * self.ILs[node][adjacentNode]))
                total_reward = total_reward + reward

                if self.upwardNetwork[node] != [0]:
                    intransit_reward = -1 * self.H[(self.upwardNetwork[adjacentNode][0], adjacentNode)] * \
                                       self.ASs[node][adjacentNode][0]

                    total_reward = total_reward + intransit_reward

                history[(adjacentNode, node)]['r'].append(reward)
            for otherNode in self.OSs[node]:
                if newDemand[0] != 0:
                    self.OSs[node][otherNode] = torch.min(currentInv, currentBackorders + newDemand) * \
                                                ((self.AOs[node][otherNode][0][0]) / newDemand[0])
                if otherNode != 0:
                    self.ASs[otherNode][node].append(self.OSs[node][otherNode])

        for node in sorted(self.upwardNetwork.keys()):
            for adjacentNode in self.upwardNetwork[node]:
                if self.t != self.T - 1:
                    history[(adjacentNode, node)]['total_r'].append(total_reward)
                else:
                    if self.downwardNetwork[node] == [0]:
                        salvageCost = -1 * torch.max(self.ILs[node][adjacentNode],
                                                     torch.zeros(len(self.ILs[node][adjacentNode])).to(
                                                         device=self.device)) * self.salvageValue
                        total_reward = total_reward + salvageCost
                    history[(adjacentNode, node)]['total_r'].append(total_reward)

        self.totalReward.append(total_reward[0])

        if self.verbosity == 2:
            print("t: ", self.t)
            print("ILs: ", [[self.ILs[i][j][0].item() for j in self.ILs[i]] for i in sorted(self.ILs.keys())])
            print("Qs: ", [[self.Qs[i][j][0].item() for j in self.ILs[i]] for i in sorted(self.Qs.keys())])
            print("reward: ", [history[i]['r'][-1][0].item() for i in history.keys()])
            print("AOs: ", [[self.AOs[i][j][0][0].item() for j in self.AOs[i]] for i in sorted(self.AOs.keys())])
            print("ASs: ", [[self.ASs[i][j][0][0].item() for j in self.ASs[i]] for i in sorted(self.ASs.keys())])
            print("OOs: ", [[self.OOs[i][j][0].item() for j in self.OOs[i]] for i in sorted(self.OOs.keys())])
            print("OSs: ", [[self.OSs[i][j][0].item() for j in self.OSs[i]] for i in sorted(self.OSs.keys())])
        self.t += 1
        if self.t == self.T:
            done = True

        return history, done, -torch.tensor(self.totalReward)

    def TakeAction(self, states, teacher):
        states = states.float().to(device=self.device)
        rawActions = teacher.AgentNet(states)
        if not isinstance(rawActions, dict):
            raise Exception("network outputs should be dicts")
        return rawActions

    def BSAction(self, s, threshold):
        s = s.float().to(device=self.device)
        threshold = threshold.float().to(device=self.device)
        return torch.max(threshold - s, torch.zeros_like(threshold))


def buildGraph(num_nodes, LO, LS, threshold_values, H_values, P_values, initializtion, T, dGen, miniBatch,
                             RLEdges, edgeStrategy, device, verbosity=0, demand_mean=5,
                             appraoch='up-to-order', salvageValue=0):
    nodes = list(range(1, num_nodes + 1))
    edges = [(i, i + 1) for i in range(num_nodes)]
    downwardNetwork = {i: [(i + 1) % (num_nodes + 1)] for i in range(1, num_nodes + 1)}
    upwardNetwork = {i: [(i - 1)] for i in range(1, num_nodes + 1)}
    threshold = dict(zip(edges, threshold_values))  # Base-stock S (threshold)
    H = dict(zip(edges, H_values))
    P = dict(zip(edges, P_values))
    return EnvironmentDirect(H, P, T, dGen, miniBatch, nodes, edges, downwardNetwork, upwardNetwork, RLEdges,
                             edgeStrategy, device, threshold, initializtion, LS, LO, verbosity, demand_mean,
                             appraoch, salvageValue)
