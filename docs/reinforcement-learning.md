# Reinforcement Learning Introduction

## Introduction

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.

## Core Concepts

### Key Components

1. **Agent**: The learner/decision maker
2. **Environment**: The world the agent interacts with
3. **State (s)**: Current situation of the agent
4. **Action (a)**: Choices available to the agent
5. **Reward (r)**: Feedback from the environment
6. **Policy (π)**: Strategy for choosing actions

### The RL Loop

```
State → Agent → Action → Environment → Reward + New State → ...
```

## Types of RL Algorithms

### 1. Value-Based Methods

Learn the value of states or state-action pairs.

**Q-Learning**
- Learn Q-values: Q(s, a)
- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**Deep Q-Network (DQN)**
- Use neural networks to approximate Q-values
- Experience replay
- Target network

**Variants:**
- Double DQN
- Dueling DQN
- Rainbow DQN

### 2. Policy-Based Methods

Directly learn the policy without value functions.

**Policy Gradient**
- REINFORCE algorithm
- Directly optimize policy parameters

**Advantages:**
- Can handle continuous action spaces
- Stochastic policies
- Better convergence properties

### 3. Actor-Critic Methods

Combine value-based and policy-based approaches.

**Popular Algorithms:**
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous A2C)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

## Exploration vs. Exploitation

### Strategies

1. **ε-Greedy**: Random action with probability ε
2. **Softmax**: Probabilistic action selection
3. **UCB**: Upper Confidence Bound
4. **Noise-based**: Add noise to actions

## Key Challenges

### Credit Assignment Problem
- Which actions led to the reward?
- Solution: TD learning, eligibility traces

### Exploration-Exploitation Tradeoff
- Explore new actions vs. exploit known good actions
- Solution: ε-greedy, UCB, intrinsic motivation

### Stability and Convergence
- Deep RL can be unstable
- Solutions: Experience replay, target networks, PPO clipping

## Popular Environments

### OpenAI Gym
```python
import gym

env = gym.make('CartPole-v1')
state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
```

### Classic Control
- CartPole
- MountainCar
- Pendulum

### Atari Games
- Breakout
- Pong
- Space Invaders

### Robotics
- MuJoCo environments
- PyBullet
- Isaac Gym

## Implementation Example

### Basic Q-Learning

```python
import numpy as np

# Initialize Q-table
Q = np.zeros([state_space_size, action_space_size])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Choose action (ε-greedy)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        
        # Update Q-value
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
```

## Advanced Topics

### Multi-Agent RL
- Cooperative agents
- Competitive agents
- Mixed scenarios

### Inverse RL
- Learn reward function from demonstrations

### Model-Based RL
- Learn environment model
- Plan using the model

### Hierarchical RL
- Learn hierarchical policies
- Options framework

### Meta-RL
- Learning to learn
- Few-shot adaptation

## Best Practices

1. **Start Simple**
   - Begin with classic control tasks
   - Use simple algorithms (DQN, A2C)

2. **Hyperparameter Tuning**
   - Learning rate is crucial
   - Discount factor affects long-term planning
   - Exploration rate decay

3. **Monitoring**
   - Track episode rewards
   - Monitor training stability
   - Visualize learned policies

4. **Reproducibility**
   - Set random seeds
   - Log hyperparameters
   - Version control code

## Tools and Libraries

### Stable Baselines3
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Ray RLlib
```python
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
```

## Applications

- **Game Playing**: Chess, Go, Atari
- **Robotics**: Manipulation, navigation
- **Autonomous Vehicles**: Path planning
- **Resource Management**: Traffic control, energy
- **Finance**: Trading strategies
- **Healthcare**: Treatment optimization

## Resources

- [Sutton & Barto - RL Book](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [DeepMind x UCL RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym](https://www.gymlibrary.dev/)
