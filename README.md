# deep-reinforcement-learning-p2-double-jointed-arms
Exploration of policy gradient and actor critic methods with a specific task from the Unity ML Agents environment involving keeping simulated arms on a target.

# Checklist

- [ ] Code for training the agent is complete with good docs
- [ ] Pytorch and Python 3 were used
- [ ] There are saved model weights for the successful agent in the repo
- [x] This file exists
- [ ] This file describes the project environment details, such as state and action spaces and when the environment is solved
- [ ] This file has instructions for getting the necessary dependencies and downloading files needed to train and use the agent
- [ ] This file has a description for how to run the agent
- [ ] I've got a Report.md outlining the implementation, including the learning algorithm used, the hyperparameters chosen, and the model architectures for ANNs
- [ ] There is a plot of the rewards in the Report.md file showing either successfully meeting version 1 or 2 of the task
- [ ] The Report.md file has the number of episodes required to solve the environment as well
- [ ] The Report.md has some concrete suggestions for how I might improve on the implementation. 

## Project Environment Details

This environment is a modified version of the Reacher Environment from Unity ML Agents. The goal for the agent or agents (depending on whether you are using the single agent version or the multiple agents version of the environment) is to learn to maneuver a double-jointed arm in such a way that the end of the arm remains within a target zone for as many time steps as possible.

State Space: For this training we will use a lower level observation space (i.e. not learning directly from pixels), consisting of a vector with 33 floating point entries. Each of entries corresponds to some state information about the mechanical arm, either relating to its position, rotation, or velocity

Action Space: The action space for the Reacher task is made up of a vector with four floats in the range [-1, 1]. These correspond to the torque you could apply to the two mechanical joints. 

Environment Considered Solved When: The reacher task is solved when the hand of the double jointed arm has remained in the goal location for an average of +30 time steps for 100 consecutive episodes.

## Project Setup, Downloading the Necessary Dependencies

### Dependencies

This task was trained using a python environment with python 3.6

1. create and activate the python environment:
    * *Linux* or *Mac*:
    ``` 
    conda create --name reacher python=3.6
    source activate reacher
    ```

    * *Windows*:
    ```
    conda create --name reacher python=3.6
    activate reacher
    ```

2. Download the environment that will be used with unity mlagents. this project was based on a course so the 
environment used is not from the unity mlagents page and instead is only available at these links:
    - [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    - [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)

extract the executable and associated dll file from the archive appropriate to your platform, and then copy it into the root of this cloned repository. The relative path from the root of your cloned version of this repo 
to the reacher executable should then be: `Reacher_Windows_x86_64/Reacher.exe`

3. install the dependencies from the requirements.txt
    ```
    pip install -r requirements.txt
    ```

4. create an ipython kernel for your conda environment so you will have access to the packages you installed, and make sure to select it when in an ipython notebook for this codebase:
    ```
    python -m ipykernel install --user --name reacher --display-name "reacher"
    ```


## Running the Agent


