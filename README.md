[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

## Environment

This project solves the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. Rewards up to +.04 are provided for each step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This project uses the Reacher version that contains 20 identical agents, each with its own environment.

The environment is considered to be solved when agents have an average score of +30 over 100 consecutive episodes.

## Installation

1. Download the environment from one of the links below.

      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Clone https://github.com/hoonji/p2_continuous-control , place the environment file the directory, and unzip (or decompress) the file. 

3. Note that the project contains old dependencies and uses an old python version. The simplest way to set this up is to use conda and install dependencies from the `udacity/Value-based-methods` repo.

```
conda create --name drlnd python=3.6
source activate drlnd
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

4. Create an IPython kernel for the drlnd conda environment

`python -m ipykernel install --user --name drlnd --display-name "drlnd"`

Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

## Instructions

Run `Continuous_Control.ipynb` to train the agent, plot the learning curve, and run agents using the last model.
