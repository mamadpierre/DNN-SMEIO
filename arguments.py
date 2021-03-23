import argparse
import json
import torch
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the Environment and RL algorithm')

    parser.add_argument(
        '--onlySim', default=False, action='store_true', help='No Cuda')

    parser.add_argument(
        '--from_scratch', default=False, action='store_true', help='Not using the saved weight')

    parser.add_argument(
        '--approach', default='OUL', help='up-to-order or quantity')

    parser.add_argument(
        '--salvageValue', type=int, default=0, help=' Salvage value for nodes close to customer ')

    parser.add_argument(
        '--verbosity', type=int, default=0,
        help='How much you want the package print for you')

    parser.add_argument(
        '--demand_Mean', type=int, default=5, help='order mean')

    parser.add_argument(
        '--demand_STD', type=int, default=1, help='STD of order')

    parser.add_argument(
        '--duplicate', type=int, default=1, help='number of duplicate runs')

    parser.add_argument(
        '--simDuplicate', type=int, default=2000, help='number of duplicate runs for simulation')

    parser.add_argument(
        '--torchSeed', type=int, default=13, help='Torch manual_seed')

    parser.add_argument(
        '--test_instances', type=int, default=100, help='test instances to average over')

    parser.add_argument(
        '--edges', nargs='+', type=int, default=[0, 1, 1, 2, 2, 3],
        help='Edges of the SCN, if we have  1 3 2 5 ---> (1,3) & (2,5)')

    parser.add_argument(
        '--h', nargs='+', type=int, default=[2, 4, 7], help='holding cost for each edge')

    parser.add_argument(
        '--p', nargs='+', type=int, default=[0, 0, 37.2], help='shortage cost for each edge')

    parser.add_argument(
        '--threshold', nargs='+', type=int, default=[10.69, 5.53, 6.49], help='shortage cost for each edge')

    parser.add_argument(
        '--DNNEdges', nargs='+', type=int, default=[0, 1, 1, 2, 2, 3], help='Edges we study, if we have  1 3 2 5 ---> (1,3) & (2,5)')

    parser.add_argument(
        '--edgeStrategy', default='DNN', help='inventory policy governing the edge we study')

    parser.add_argument(
        '--device', default='cpu', help='cpu or cuda')

    parser.add_argument(
        '--numpySeed', type=int, default=43, help='Random seed')

    parser.add_argument(
        '--episode', type=int, default=30000, help='Number of episodes')

    parser.add_argument(
        '--T', type=int, default=10, help='Episode horizon')

    parser.add_argument(
        '--miniBatch', type=int, default=2, help='Mini-batch size')

    parser.add_argument(
        '--lr', type=float, default=0.01, help='Optimization algorithm learning rate')

    parser.add_argument(
        '--NetworkPATH', default='SavedWeights.pt', help='Saving path')

    parser.add_argument(
        '--neuralNodes', nargs='+', type=int, default=[24, 24, 24, 24],
        help='nodes in hidden layers (we excluded first and last layers cause they are fixed')

    parser.add_argument(
        '--sharedLayers', type=int, default=1, help='number of shared layers')

    parser.add_argument(
        '--LO', nargs='+', type=int, default=[0, 0, 0],
        help='order leadtime')

    parser.add_argument(
        '--LS', nargs='+', type=int, default=[2, 1, 1],
        help='Shipment leadtime')

    parser.add_argument(
        '--activation', default='softplus', help='it can be softplus or ReLU or leakyReLU')

    parser.add_argument(
        '--noCuda', action='store_true', help='No Cuda')

    parser.add_argument(
        '--noTest', default=False, action='store_true', help='No test')

    args = parser.parse_args()

    return args


def cudaAvailability(no_cuda, torchSeed):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(torchSeed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(0)
    return device


def prepareEdges(edges):
    if len(edges) % 2 != 0:
        raise Exception(" The nodes connecting edges should be of even length")
    preparedEdges = []
    for edge in range(0, len(edges), 2):
        preparedEdges.append(tuple([edges[edge], edges[edge + 1]]))
    return preparedEdges




def prepareNeuralNetStructure(iinput, nodes, activation):
    layers = []
    layers.append((iinput, nodes[0]))
    for i in range(len(nodes)):
        if i != len(nodes) - 1:
            layers.append((nodes[i], nodes[i + 1]))
    layers.append((nodes[-1], 1))
    if activation == "softplus":
        Activation = F.softplus
    if activation == "ReLU":
        Activation = F.relu
    if activation == "leakyReLU":
        Activation = F.leaky_relu

    return layers, Activation





if __name__ == "__main__":
    args = get_args()







