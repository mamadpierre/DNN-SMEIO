# DNN-SMEIO
DNN-SMEIO

This python package contains the implementation of the DNN-SMEIO algorithm for a serial SCN structure discussed in the paper, "Simultaneous Decision Making for Stochastic Multi-echelon Inventory Optimization with Deep Neural Networks as Decision Makers." 

```
requirements:

Pytorch >= 1.5
numpy >= 1.15

```

To run the code with defualt arguments use

```
python main.py --episode 500

```

The above code executes a DNN-SMEIO algorithm for the case (3) of the serial cases and only for 500 episodes. Because this command by default considers pre-trained weights. In order to train the networks from scratch you can add the option ``` --from_scratch```. The algorithm should reach a similar result with around 20k episodes.
For other serial structures one should modify arguments related to the initialization of those SCNs such as ```h, p, LS, LO threshold, ...```. 

In order to only run a simulation with already defined OULs, you can use the option ```--onlySim``` and further modify the number of the independent runs by ```--simDuplicate```. If you use the code, please cite its corresponding paper:

```
@article{pirhooshyaran2020simultaneous,
  title={Simultaneous Decision Making for Stochastic Multi-echelon Inventory Optimization with Deep Neural Networks as Decision Makers},
  author={Pirhooshyaran, Mohammad and Snyder, Lawrence V},
  journal={arXiv preprint arXiv:2006.05608},
  year={2020}
}
```





