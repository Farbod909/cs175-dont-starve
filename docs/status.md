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
At the moment we are using Deep Q-Learning with a feed-foward neural network.
Our feed forward neural network takes a one-hot encoding of every farm cell. 
It contains 5 hidden layers that contain 15 weights each with ReLU as the activation function at the output of every layer with the exception of one that has a tanh function. 
It then outputs the approximated Q-value for each action from the next state. 

![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/Feed_Forward_Graphic.png)

## Evaluation
In our project home page we described the following: 

  - Wheat needs to be planted in 1-wide rows or columns, with at least 2 wheat per line.
  - Potatoes need to be planted with two other potatoes nearby, in an L shape, or with three other potatoes.
  - Carrots will only grow if they have zero or one nearby carrots.
  - Beetroots require four or more adjacent beetroots, including diagonals.

Quantitatively, a strong performing agent is one that is able to reach a reward close to or equal to the number of cells in the farm; meaning that a crop grew in almost every cell.

If it performs well according to the quantitative metric, we should see patterns according to the planting restrictions we described above.
For example, if it plants wheat in rows, then we know it is learning successfully. 
If it continues to act random, on the other hand, our model is not learning properly.

## Video

## Remaining Goals and Challenges
We are currently working on applying a convolutional neural network (CNN) to our task.
Considering a CNN can have multiple filters to scan over the farm, it may have the potential of recognizing patterns that lead to high reward such as rows of wheat.
It will be interesting to visualize the filters in order to confirm that they are picking up these patterns.




## Resources Used
http://deeplizard.com/learn/video/0bt0SjbS3xc
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0



#------ find different placement for this -----
The environment is deterministic (a square farm of a given size), so the agent does not need any percepts while planting, only to remember what crops it has planted and where. This also makes the agent very fast, since waiting for Malmo's percepts such as a grid view is slow and often unreliable. After the agent is done planting, it harvests the crops and counts them to find its reward for that iteration.
