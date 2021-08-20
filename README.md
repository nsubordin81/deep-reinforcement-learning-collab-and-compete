# deep-reinforcement-learning-p3-collaboration-and-competition
training a DRL algorithm in a multi agent environment

# Checklist

- [x] Code for training the agent is complete with good docs
- [x] Pytorch and Python 3 were used
- [x] There are saved model weights for the successful agent in the repo
- [x] This README file exists
- [x] This file describes the project environment details, such as state and action spaces and when the environment is solved
- [x] This file has instructions for getting the necessary dependencies and downloading files needed to train and use the agent
- [x] This file has a description for how to run the agent
- [x] I've got a Report.md outlining the implementation, including the learning algorithm used, the hyperparameters chosen, and the model architectures for ANNs
- [x] There is a plot of the rewards in the Report.md file showing an agent successfully completing the task
- [x] The Report.md file has the number of episodes required to solve the environment as well
- [x] The Report.md has some concrete suggestions for how I might improve on the implementation. 

## Project Environment Details

This environment is a modified version of the Tennis Environment from Unity ML Agents. This is a cooperative multi-agent task in which both agents are attempting to keep a ball in play by controlling their respective paddles, and ultimately get the ball over the net as many times as possible without it hitting the ground or going out of bounds. The reward dynamic is given by a +0.1 for each successful time hitting the ball over the net and -0.1 each time the ball goes out of bounds or hits the ground.

State Space: the state or observation space is made up of 8 variables that provide the agents information about the position and velocity of the ball and racquet, and since this is a multi agent space each agent gets their own version of this 8 variable vector stacked 3 deep for a 3x8=24, so the vector of observations is length 24. The complete state information would be 2 of these such vectors since there are 2 agents.

Action Space: the action space is composed of a 2 variable vector where one of the variables corresponds to the paddle's forward and backward movement wrt the net and the other corresponds to a 'jump' action that the paddle can perform, moving up off of the table (presumably to collide with the ball in some way).

Environment Considered Solved When: This environment is considered solved when the two agents are able to keep up a volleys long enough that the score accumulated by the highest performing agent per episode averages out to +0.5 over 100 consecutive episodes. 

## Project Setup, Downloading the Necessary Dependencies

### Dependencies

This task was trained using a python environment with python 3.6

First step is to clone this repository down to your machine, and from there:

1. create and activate the python environment:
    * *Linux* or *Mac*:
    ``` 
    conda create --name tennis python=3.6
    conda activate tennis
    ```

    * *Windows*:
    ```
    conda create --name tennis python=3.6
    conda activate tennis
    ```

2. Download the environment that will be used with unity mlagents. this project was based on a course so the 
environment used is not from the unity mlagents page and instead is only available at these links:
    - [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)

extract the executable and associated dll file from the archive appropriate to your platform, and then copy it into the root of this cloned repository. The relative path from the root of your cloned version of this repo 
to the tennis executable should then be: `Tennis_Windows_x86_64/Tennis.exe` for the Windows example

3. install the dependencies from the requirements.txt
    ```
    pip install -r requirements.txt
    ```

4. create an ipython kernel for your conda environment so you will have access to the packages you installed, and make sure to select it when in an ipython notebook for this codebase:
    ```
    python -m ipykernel install --user --name tennis --display-name "tennis"
    ```


## Running the Agent

The code to train and run the agent are in MADDPG.ipynb jupyter notebook. If you've installed the conda environment above and created a kernel for it, you should be able to start a notebook instance from your terminal running the command `jupyter notebook ./SPDDPG.ipynb` then, 

### Running the Agent: 

1. run first 2 cells, labeled 'import the necessary packages' and 'instantiate the environment and agent'
2. run the final cell, labeled 'Watch a smart agent!'

### Retraining the Agent: 

1. run first 2 cells, labeled 'import the necessary packages' and 'instantiate the environment and agent'
2. run the cell labeled 'Train the Agent with DDPG'

note - if you want to train the agent and then run it, in between these two steps you will need to restart the kernel just because of an issue with the unity environment, I've had issues closing it and then instantiating it again within the same kernel, so restarting the kernel resets any lingering dependency unity has created by launching the environment.
