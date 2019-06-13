## Project Summary

While our main goal is still to create an agent that can learn to farm, we have decided to change the agent's reason for farming. Initially, we had the idea that we could make an agent motivated by in-game hunger, but that turned out to not be an interesting challenge since hunger is very slow to deplete and growing enough food to survive is fairly easy.

Our new idea is to require (with a mod) that crops be planted in certain configurations in order to grow at all. Our agent would then have to learn how to plant these crops to make efficient use of its limited farming space.

## Approach
MDP:
Our MDP is one in which every node is a cell in the farm and every outgoing edge is a potential crop it can plant. 
Reward is only given in the final state when every cell is full and we count the amount of crops that grew. 
Though reward seems scarce, we believe it is okay because the agent will always reach the terminating state because it moves throughout the farm deterministically.

Reward:
In our environment, an episode starts with the agent placed at the top-leftmost cell of an empty dirt farm. 
The agent moves through row-wise and plants a crop at every cell. 
The episode ends once the agent plants a crop at the bottom-rightmost cell.
From there, crops are given time to grow and a reward of +1 is given per crop that grew successfully. 
The max reward possible per episode is the number of cells in the farm. 

Neural Functions Approximator:
At the moment we are using Deep Q-Learning with a feed-foward neural network and replay memory.
Our feed forward neural network takes a one-hot encoding of every farm cell in batches of 10. 
It contains 5 hidden layers that contain 15 weights each with ReLU as the activation function at the output of every layer with the exception of one that has a tanh function. 
It then outputs the approximated Q-value for each action from the next state. 

![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/Feed_Forward_Graphic.png)

Our loss function is the Mean-Squared error between the approximated target value with the value of the current state-action pair.
Our approximated target value is found via the highest expected return of next state-action pair.

![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/loss_ftn.PNG)

Lastly, we utilize the epsilon-greedy algorithm with a decaying exploration rate of 0.5 - 0.2.

## Evaluation
In our project home page we described the following: 

  - Wheat needs to be planted in 1-wide rows or columns, with at least 2 wheat per line.
  - Potatoes need to be planted with two other potatoes nearby, in an L shape, or with three other potatoes.
  - Carrots will only grow if they have zero or one nearby carrots.
  - Beetroots require four or more adjacent beetroots, including diagonals.

Quantitatively, a strong performing agent is one that is able to reach a reward close to or equal to the number of cells in the farm; meaning that a crop grew in almost every cell.

We trained our model over 1000 episodes. 
The performance of our model can be seen in the figure below where the x-axis is the episode and the y-axis is the reward obtained. 
We received an average reward of 1.93; .78% short of our goal.

![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/Reward_per_episode.png)

Qualitiatively, our agent acts according to the results described above. 
The crops planted are basically random and there are no signs of patterns. 
Though this could be due to not having enough training data or a complex enough model, we believe a feed forward network in general is not complex enough to approximate our state-space.

## Video

https://www.youtube.com/watch?v=P9kLdlbPS_U&feature=youtu.be

## Remaining Goals and Challenges
A model that may be complex enough to fit our state-space is a convolutional neural network.
Considering a CNN can have multiple filters scanning over the farm, it may have the potential of recognizing patterns that lead to high reward such as rows of wheat.
It will also be interesting to visualize the filters in order to confirm that they are picking up these patterns.

Our goal for the remaining 3-weeks is to implement the CNN and fit it not only to a 2-crop scenario, but also to a 4-crop scenario.




## Resources Used
http://deeplizard.com/learn/video/0bt0SjbS3xc

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training

https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
