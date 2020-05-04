!apt-get install -y xvfb python-opengl x11-utils > /dev/null 2>&1
!pip install gym pyvirtualdisplay scikit-video > /dev/null 2>&1

%tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import base64, io, time, gym
import IPython, functools
import matplotlib.pyplot as plt
from tqdm import tqdm

!pip install mitdeeplearning
import mitdeeplearning as mdl

### Instantiate the Cartpole environment 
env = gym.make("CartPole-v0")
env.seed(1)
n_observations = env.observation_space
print("Environment has observation space =", n_observations)
n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from =", n_actions)

### Cartpole agent

# Defines a feed-forward neural network
def create_cartpole_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=n_actions, activation=None) 
  ])
  return model

cartpole_model = create_cartpole_model()

### Agent's action function 
def choose_action(model, observation):
  observation = np.expand_dims(observation, axis=0)
  logits = model.predict(observation)
  prob_weights = tf.nn.softmax(logits).numpy()
  action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0] 
  return action
  
### Agent Memory 
class Memory:
  def __init__(self): 
      self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
      self.observations = []
      self.actions = []
      self.rewards = []

  def add_to_memory(self, new_observation, new_action, new_reward): 
      self.observations.append(new_observation)
      self.actions.append(new_action)
      self.rewards.append(new_reward) 
        
memory = Memory()  


### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  return x.astype(np.float32)
  
def discount_rewards(rewards, gamma=0.95): 
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R
      
  return normalize(discounted_rewards)


### Loss function ###
# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
def compute_loss(logits, actions, rewards): 
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)  
  loss = tf.reduce_mean( neg_logprob * rewards ) 
  return loss
  
### Training step (forward and backpropagation) 

def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      # Forward propagate through the agent network
      logits = model(observations)
      loss = compute_loss(logits, actions, discounted_rewards) 
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
### Cartpole training

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
cartpole_model = create_cartpole_model()

# to track our progress
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for i_episode in range(500):

  plotter.plot(smoothed_reward.get())

  # Restart the environment
  observation = env.reset()
  memory.clear()

  while True:
      action = choose_action(cartpole_model, observation)
      next_observation, reward, done, info = env.step(action)
      memory.add_to_memory(observation, action, reward)
      
      if done:
          total_reward = sum(memory.rewards)
          smoothed_reward.append(total_reward)
          
          train_step(cartpole_model, optimizer, 
                     observations=np.vstack(memory.observations),
                     actions=np.array(memory.actions),
                     discounted_rewards = discount_rewards(memory.rewards))
          
          # reset the memory
          memory.clear()
          break
      # update our observatons
      observation = next_observation
  
saved_cartpole = mdl.lab3.save_video_of_model(cartpole_model, "CartPole-v0")
mdl.lab3.play_video(saved_cartpole)
  
