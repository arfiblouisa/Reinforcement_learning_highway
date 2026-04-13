# Reinforcement Learning project

The aim of this project is to implement different RL techniques to improve the policy of an autonomous car. We used the environment *highway-v0* from *highway_env*, usually used for decision-making in Autonomous Driving. The project is divided into two tasks : a core task, and an extension.  

The core tasks consists of training and evaluating different RL models on a shared simple configuration. Then we complexify the problem, and want our vehicle to drive in a safe way and therefore not crash in a trafic configuration with a high vehicles density.

## Settings

In order to evaluate and compare our models across this repository as well with our classmates', we are using the same configuration provided in class material. This configuration defines useful elements such as observation, rewards and environment parameters. The configuration is then changed in the extension task in order to reach better results.


### Installation

Install with pip :

```
pip install requirements.txt
```

### Repository description

This repository is organized as follows:

```text
Reinforcement_learning_highway/
├── core_task/
│   ├── Checkpoints/ # Checkpoints for the DQN 1 and 2 layers, and for the DQN trained with stable baselines
│   ├── Double DQN/  # Notebook for training the double DQN model and checkpoint
│   ├── DQN 1 Layer/ # Notebook for training a DQN model with 1 layer
│   ├── DQN 2 Layers/ # Notebook for training a DQN model with 2 layers
│   ├── Stable baselines/ #Notebook for training a DQN with stable baselines
│   ├── videos/ # Videos generated in the compare_models file to visualize an episode
│   └── compare_models.ipynb # Notebook to compare the models trained with the files above

│   
├── extension_task/
│   ├── extension_reward/
│   |
│   └── social_attention/
│   │   ├── out/                  # Training results of 3 configs. The model in saved_models is used for testing
│   │   ├── rl-agents/            # code to train a double DQN attention network
│   │   ├── runs/                 # videos from the testing
│   │   ├── social_attention.ipynb  # notebook used to run the training and testing pipeline of the social attention network
│   │   └── ...
│
├── requirements.txt      
├── README.md              
└── .gitignore
```

## 1. Core task

In this part, we train and evaluate different models on the follwoing configuration :

```
SHARED_CORE_ENV_ID = "highway-v0"

SHARED_CORE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [20, 25, 30],
    },
    "lanes_count": 4,
    "vehicles_count": 45,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "collision_reward": -1.5,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.7,
    "lane_change_reward": -0.02,
    "reward_speed_range": [22, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}
```

We decided to implement 2 DQN ourselves : one with 1 layer, and another one with 2 layers, a DQN with stable baselines, and a double DQN. In order to compare our models with a baseline, we also implemented a random model (at each time step, a random action among the list of possible actions is chosen). 

The code used for the DQN is based on the code provided in our class material while the one used for the double DQN is based on the one found in this [GitHub repository](https://github.com/eleurent/highway-env).

In the `compare_models.ipynb` notebook, we evaluated each agent on the same configuration. The DQN that we implemented ourselves performs the best on this config, and remarkably never crashes during the 50 evaluation runs. However, when we increase the `vehicles_density` in the config, all the models previously develepod now crash very often and achieve consequently a very low mean reward. That is why we chose to work on an extension task which is trying to obtain a safer driving (with less crashes) in a dense traffic.
## 2. Extension task

As explained before, the goal of the extension task is to obtain a safer driving in a dense traffic. The previous policies do not generalize well to dense traffic, as the vehicle crash very quickly. We want to improve the distance traveled by the vehicle. Two different methods were implemented :
- modifying the rewards to change the conduct style;
- using the social attention network developed by [Leurent and Mercat, 2019](https://arxiv.org/abs/1911.12250) that is supposed to be adapted to dense traffic.

The configuration used to train and test the networks is partly:
{
    "lanes_count": 4,
    "vehicles_count": 50,
    "vehicles_density": 2
}

### 2.1 Changing the rewards

### 2.2 Social attention network

The full configuration used to train the network can be found in  
*`extension_task/social_attention/rl-agents/scripts/configs/HighwayEnv/env_obs_attention_with_traffic.json`*.

The *ego architecture* described in [Leurent and Mercat, 2019](https://arxiv.org/abs/1911.12250) is used.

Therefore, a state consists of:

$$
s = (s_i)_{i \in [0, N]}, \quad \text{where} \quad
s_i =
\begin{bmatrix}
x_i \\
y_i \\
v_{x,i} \\
v_{y,i} \\
\cos(\theta_i) \\
\sin(\theta_i)
\end{bmatrix}
$$

with $s_0$ the state of the controlled vehicle, and $N$ the number of vehicles that the ego vehicle can see (we chose 15 in this simulation).

Two attention heads were used; they can be visualized during inference.

A simulation example obtained with our trained model is shown below:

<p align="center">
  <img src="extension_task/social_attention/runs/videos/demo_results_attention.gif" width="400">
</p>


## Results and Conclusions

This repository was created and equally contributed to by :
- Louisa Arfib : [https://github.com/arfiblouisa](https://github.com/arfiblouisa)
- Manon Arfib : [https://github.com/manonarfib](https://github.com/manonarfib)
- Florian De Boni : [https://github.com/FlorianDeBoni](https://github.com/FlorianDeBoni)
- Nathan Morin : [https://github.com/Nathan9842](https://github.com/Nathan9842)

