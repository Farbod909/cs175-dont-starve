## Project Summary
Our environment is a simple grid farm that is ready for planting.
Our agent's task is to determine which plants to crop at each particular cell of the farm.
The challenge, however is that crops are required to be planted in certain configurations in order to grow at all. 
Our agent would then have to learn how to plant these crops to make efficient use of its limited farming space.

The required configurations are as follows:
  - Wheat needs to be planted in 1-wide rows or columns, with at least 2 wheat per line.
  - Potatoes need to be planted with two other potatoes nearby, in an L shape, or with three other potatoes.
  - Carrots will only grow if they have zero or one nearby carrots.
  - Beetroots require four or more adjacent beetroots, including diagonals.

With a small farm such as a 3x3 farm with only 2 crops in the agent's inventory, it is possible to apply tabular Q-learning as there are a total of 2^9 (512) possible states (2 crops per cell in the farm). The tabular method, however, is not applicable when using all the crops as that gives us 4^9 (262,144). Furthermore, a 3x3 is too small to even grow a Beetroot so a farm of at least 4x4 is needed. Using all the crops, our task has, at minimum, 4^16 (4,294,967,296) possible states.

## Approaches
Considering the large number of states, we believe Deep Q-Learning is necessary to solve our task. In our implementation we attempt to solve the small 2^9 problem described above using a feed-forward neural network as well as a convolutional neural network before taking on the 4^9 farm. Particularly, for each network, we iteratively update the Q-values by sampling from replay memory at the end of every episode. The details for each network are as follows:

### Feed Forward Neural Network

### Convolutional Neural Network (CNN)
Our CNN takes as input a matrix where each cell contains an integer that represents the crop planted there and the agents position. There are two convolutional layers and 3 linear layers. The specifications for each is described in the figure below: 

$k = P(a|b)$
