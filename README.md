# Multi-Agent-Tennis
This repository is my implementation for the third project of Udacity's Deep Reinforcement Learning Nanodegree. The GIF below shows the performance of the best model which achieved a test mean score of `1.81` and the environment was solved at episode **1774**. More details regarding the architecture of the model, its best hyperparameters, results, and future work can be found in [this report](./Report.pdf).
![Trained Agent](./media/best_model.gif)

## 💡 Project Details
This project uses Unity's ML Agents tennis environment which is a multi-agent environment. In this environment, two agents control rackets to bounce a ball over a net. 

If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## 👩🏻‍💻 Getting Started
To set up your python environment to run the code in this repository, follow the instructions below. 

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the DRL repository, and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
5. Clone this repository to be able to run its code and use its models.
    ```bash
    git clone https://github.com/Iten-No-404/Multi-Agent-Tennis.git
    ```

Note that this installation guildline is adapted from [Udacity's DRL GitHub repository](https://github.com/udacity/deep-reinforcement-learning)

## 🧭 How to use?

Disclaimer, some of this code is adapted from the official [Udacity's DRL GitHub repository](https://github.com/udacity/deep-reinforcement-learning).

The repository is structured as follows:
- [Tennis.ipynb](./Tennis.ipynb) contains an introduction to the Banana Navigation environment and how to use it with Unity's ML Agents.
- [model.py](./model.py) defines the Actor and Critic network architecture.
- [ddpg_agent.py](./ddpg_agent.py) defines the DDPG agent used as well as the ReplayBuffer class for handling the experience replay buffer logic. It also has a class for the Ornstein Uhlenbeck Action Noise process (OUNoise) needed for a more realistic portrayal of the actions. 
- [multi_ddpg_agent.py](./multi_ddpg_agent.py) defines the DDPG multi-agent class. It contains a single Critic Network and multiple Actor Networks; one for each agent. All the networks share the same is experience replay buffer.
- [utils.py](./utils.py) contains a small helping function for creating ordered directories.
- [Tennis.ipynb](./Tennis.ipynb) contains the training loop and the 3 DDPG hyperparameter attempts, and a simple loop to test the trained models and measure their mean scores over 100 episodes.
- [Play_Best_Model.ipynb](./Play_Best_Model.ipynb) contains a simple loop to run the best model on the environment to view its performance.
- [Report.pdf](./Report.pdf) describes the used learning algorithm, the best hyperparameters, results and future work.
- [The ddpg_trials](.ddpg_trials) directory includes the results of all 3 training trials. Each subfolder indicates a trial and contain the following:
  - [critic.pth](./ddpg_trials/3/critic.pth) which are the saved weights of the Critic model after training.
  - [0_actor.pth](./ddpg_trials/3/0_actor.pth) which are the saved weights of the first Actor model after training.
  - [1_actor.pth](./ddpg_trials/3/1_actor.pth) which are the saved weights of the second Actor model after training.
  - [parameters.json](./ddpg_trials/3/parameters.json) which contains the hyperparameters used in training this model.
  - [scores.json](./ddpg_trials/3/scores.json) contains all the scores (i.e. rewards) achieved in each training episode.
  - [test_scores.json](./ddpg_trials/3/test_scores.json) contains all the scores (i.e. rewards) achieved in each testing episode (from the 100 test episodes).
  
Note that the above links are for the best model whose subfolder is `3`.
