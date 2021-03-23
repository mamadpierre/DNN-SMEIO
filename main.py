from __future__ import print_function, division
import numpy as np
from arguments import prepareNeuralNetStructure, get_args, prepareEdges
from utils import Run, SCNInit, sim
from environment import buildGraph
import torch




def main(args):
    args.DNNEdges = prepareEdges(args.DNNEdges)
    if not args.onlySim:
        distGenerator, threshold_values, H_values, P_values, LO, LS, initializtion = SCNInit(args)

        myEnvironment = buildGraph(3, LO, LS, threshold_values, H_values, P_values, initializtion, args.T
                                   , distGenerator, args.miniBatch, args.DNNEdges, args.edgeStrategy, args.device,
                                   args.verbosity, args.demand_Mean, args.approach, args.salvageValue)
        print("SCN nodes: ", myEnvironment.nodes, "SCN edges: ", myEnvironment.edges,
              "thresholds: ", myEnvironment.threshold, "DNNEdges: ", myEnvironment.DNNEdges, "LS: ", myEnvironment.LS,
              "LO: ", myEnvironment.LO)
        layers, Activation = prepareNeuralNetStructure(len(myEnvironment.edges), args.neuralNodes, args.activation)
        RunResultCompact, RunResultWhole, bestModel = Run(myEnvironment, layers, Activation, args.device, args)
        threshold_values = torch.tensor([RunResultCompact[1][args.DNNEdges[i]] for i in range(len(args.DNNEdges))])
        sim(args, threshold_values)
    else:
        threshold_values = torch.tensor(args.threshold)
        sim(args, threshold_values)

if __name__ == "__main__":
    main(get_args())
