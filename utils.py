import torch
import numpy as np
from AgentNet import Teacher
from arguments import prepareEdges
from environment import buildGraph


def SCNInit(args):
    distGenerator = lambda: args.demand_Mean + torch.randn(args.miniBatch) * args.demand_STD
    threshold_values = torch.tensor(args.threshold)
    H_values = np.array(args.h)
    P_values = np.array(args.p)
    LO = {i + 1: [args.LO[i]] for i in range(len(args.LO))}
    LS = {i + 1: [args.LS[i]] for i in range(len(args.LS))}

    initializtion = {1: [args.demand_Mean * args.LS[0]], 2: [args.demand_Mean * args.LS[1]],
                     3: [args.demand_Mean * args.LS[2]]}

    return distGenerator, threshold_values, H_values, P_values, LO, LS, initializtion




def rollout_trajectory(environment, teacher):
    history = {}
    for edge in environment.edges:
        history[edge] = {'s': [], 'a': [], 'r': [], 'total_r': [], 'ns': []}
    done = False
    environment.reset()
    while not done:
        history, done, _ = environment.step(teacher, history)
        if done:
            break
    return history


def Run(myEnvironment, layers, Activation, device, args):
    runBestResultCost = float('Infinity')
    for instance in range(args.duplicate):
        myTeacher = Teacher(layers, args.sharedLayers, Activation, device, args.lr, args.DNNEdges)
        if not args.from_scratch:
            Checkpoint = torch.load("SavedWeights.pt")
            myTeacher.AgentNet.load_state_dict(Checkpoint['model_state_dict'])
            myTeacher.optim.load_state_dict(Checkpoint['optimizer_state_dict'])
            print("Network loaded")

        instanceBestResult, instanceWholeResult = trainTest(myEnvironment, myTeacher, args.DNNEdges, args.episode,
                                                                  args.noTest,
                                                                  args.test_instances,
                                                                  args.NetworkPATH)
        if instanceBestResult[0][args.DNNEdges[0]] < runBestResultCost:
            runBestResult = instanceBestResult
            runBestResultCost = instanceBestResult[0][args.DNNEdges[0]]
            runWholeBestResult = instanceWholeResult
            bestModel = '/Actor' + str(instance) + '.pt'
    return runBestResult, runWholeBestResult, bestModel


def trainTest(Environment, Teacher, DNNEdges, episode, noTest, test_instances, path):
    if not noTest:
        rewardResult = {}
        rew_lst = {}
        netOut = {}
        netOutSeparate = {}
        netResult = {}
        reward_plot = {}
        netOut_plot = {}
        netOut_plotSeparete = {}
        for edge in DNNEdges:
            rewardResult[edge] = []
            reward_plot[edge] = []
            netOut_plot[edge] = []
            netResult[edge] = []
            netOut_plotSeparete[edge] = []

    for omega in range(episode):
        s_tot = 0
        history_t_whole = rollout_trajectory(Environment, Teacher)
        for edge in DNNEdges:
            history_t = history_t_whole[edge]
            rewards = torch.stack(history_t['total_r'])
            policy_loss = -1 * torch.mean(rewards)
            s_tot = s_tot + policy_loss
        Teacher.optim.zero_grad()
        s_tot.backward()
        Teacher.optim.step()

        if (omega % 100 == 0 or omega == episode - 1) and not noTest:
            for edge in DNNEdges:
                rew_lst[edge] = []
                netOut[edge] = []
                netOutSeparate[edge] = []
            for _ in range(test_instances):
                test_history_whole = rollout_trajectory(Environment, Teacher)
                for edge in DNNEdges:
                    test_history = test_history_whole[edge]
                    rew_lst[edge].append(torch.mean(torch.stack(test_history['total_r'])))
                    netOutSeparate[edge].append(torch.stack(test_history['a']))
                    netOut[edge].append(torch.mean(torch.stack(test_history['a'])))
            best_so_far = {}
            for edge in DNNEdges:
                reward_plot[edge].append(torch.mean(torch.stack(rew_lst[edge])).detach().cpu().numpy().item())
                netOut_plot[edge].append(torch.mean(torch.stack(netOut[edge])).detach().cpu().numpy().item())
                netOut_plotSeparete[edge].append(
                    torch.mean(torch.stack(netOutSeparate[edge]), 0).detach().cpu().numpy())
                index = reward_plot[edge].index(np.max(reward_plot[edge]))
                best_so_far[edge] = round(netOut_plot[edge][index], 2)
                if index == len(reward_plot[edge]) - 1:
                    if edge == DNNEdges[-1]:
                        print("new best cost:", -reward_plot[edge][index], "new best OULs: ", best_so_far)
                        print("saving the network")
                        torch.save({'model_state_dict': Teacher.AgentNet.state_dict(), 'optimizer_state_dict':
                            Teacher.optim.state_dict()}, path)

        for edge in DNNEdges:
            rewardResult[edge] = -reward_plot[edge][index]
            netResult[edge] = netOut_plot[edge][index]
    if not noTest:
        return [rewardResult, netResult], [reward_plot, netOut_plot, netOut_plotSeparete]
    else:
        return 0, 0


def sim(args, threshold_values):
    print("-"*50)
    print("start of simulation")
    args.edgeStrategy = 'notDNN'
    distGenerator, _, H_values, P_values, LO, LS, initializtion = SCNInit(args)
    reward = 0
    for _ in range(args.simDuplicate):
        args.numpySeed = np.random.randint(1, 20000)
        args.torchSeed = np.random.randint(1, 20000)
        myEnvironment = buildGraph(3, LO, LS, threshold_values, H_values, P_values, initializtion, args.T
                                   , distGenerator, args.miniBatch, args.DNNEdges, args.edgeStrategy, args.device,
                                   args.verbosity, args.demand_Mean, args.approach, args.salvageValue)

        history = {}
        for edge in myEnvironment.edges:
            history[edge] = {'s': [], 'a': [], 'r': [], 'total_r': []}
        for _ in range(args.T):
            history, _, totalReward = myEnvironment.step(None, history)
        if isinstance(totalReward, torch.Tensor):
            reward += np.mean(totalReward.cpu().numpy())
        else:
            reward += np.mean(totalReward)
    Numeric_cost = reward / args.simDuplicate
    for i in range(len(args.DNNEdges)):
        print("edge: ", args.DNNEdges[i], "OUL: ", round(threshold_values[i].item(),2))
    print("simulation cost: ", round(Numeric_cost, 2))
