# Reinforcement Learning project

The aim of this project is to implement different RL techniques to improve the policy of an autonomous car. We used the environment *highway-v0* from *highway_env*, usually used for decision-making in Autonomous Driving. The project is divided into two tasks : a core task, and an extension.  

Expliquer très rapidment le but des core et extension tasks.  
Mettre des photos/vidéos des simulations et de l'environnement

## Settings

TO WRITE

### Installation

Install with pip :

```
pip install requirements.txt
```
FAIRE LES REQUIREMENTS !! S'assurer que tourne bien sur eleurent

### Repository description

This repository is organized as follows:

```text
Reinforcement_learning_highway/
├── core_task/
│   ├── convlstm/                 # Checkpoints for the ConvLSTM model according to the loss used during training
│   │   ├── advanced_torrential/
│   │   ├── mse/
│   │   └── ...
│   └── unet/                     # Checkpoint for the U-Net model corresponding to training with MSE loss
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

Rappeler en quoi ça consiste
- DQN
- Stable baseline
- Double DQN
- random
- untrained (à voir si on garde)

## 2. Extension task

Rappeler la config

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
  <video width="400" controls>
    <source src="extension_task/social_attention/runs/videos/rl-video-episode-4.mp4" type="video/mp4">
  </video>
</p>


## Results and Conclusions

This repository was created and equally contributed to by :
- Louisa Arfib : [https://github.com/arfiblouisa](https://github.com/arfiblouisa)
- Manon Arfib : [https://github.com/manonarfib](https://github.com/manonarfib)
- Florian De Boni : [https://github.com/FlorianDeBoni](https://github.com/FlorianDeBoni)
- Nathan Morin : [https://github.com/Nathan9842](https://github.com/Nathan9842)

