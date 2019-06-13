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

With a small farm such as a 3x3 farm with only 2 crops in the agent's inventory, it is possible to apply tabular Q-learning as there are a total of 2^9 (512) possible states (2 crops per cell in the farm). 
The tabular method, however, is not applicable when using all the crops as that gives us 4^9 (262,144). 
Furthermore, a 3x3 is too small to even grow a Beetroot so a farm of at least 4x4 is needed. 
Using all the crops, our task has, at minimum, 4^16 (4,294,967,296) possible states.

## Approaches
Considering the large number of states, we believe Deep Q-Learning is necessary to solve our task. 
In our implementation we attempt to solve the small 2^9 problem described above using a feed-forward neural network as well as a convolutional neural network before taking on the 4^9 farm. 
Particularly, for each network, we iteratively update the Q-values by sampling from replay memory at the end of every episode. 

### Baseline
Our main metric of performance is average reward per episode.
We can obtain a baseline of performance by observing how an algorithm that plants crops randomly performs under the constraints we set.

![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/figures/avg_random_reward.PNG)

### Feed Forward Neural Network
In our previous report, we trained a feed-foward neural network with 5 hidden layers & 15 neurons per layer on our farm over 1000 iterations. 
We did not find that it was effective approach, but decided to make the network bigger and run it over 10,000 iterations to obtain stronger evidence that this type of network can't fit our problem set. 
The average reward for the larger network with more training data is shown in the figure below:

![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/figures/r-list-10k-avg.png)

Regardless of the fact that we gave it 10x more data and increased the number of layers and nodes, a feed forward neural network does not seem to be complex enough to fit our problem set. 
As training progresses, it begins to do worse than random.

### Convolutional Neural Network (CNN)
CNNs are best known for their high performance in machine vision as the filters in the convolution layers aid it in recognizing patterns amongst the pixels. 
Considering our task is to find plant formations that lead to maximum crop growth, a CNN has potential to fit the problem set if we translate the farm into a format that's similar to single-channel images.
We do this by assigning every crop an ID and passing in our farm as a 3d matrix of the shape [1,3,3].
This matrix is then passed to our CNN that holds 2 convolutional layers and 3 linear layers.

![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/figures/Example%20Input.PNG)

![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/figures/cropped_cnn_fig.png)


We trained our network using greedy-epsilon with epsilon set to .5. Plotting our average reward as we train with, we find that the convolutional neural network does significantly better than the feed-forward neural network. 
As we reach 10,000 episodes, the average reward starts reaching that of the random baseline. 
It is, however, difficult to tell if it will do better than random and can only be proved by training the network over more episodes.
![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/figures/2_crop_avg_reward.PNG)

To test our CNN we load the trained model and set exploration to 0.
Unfortunately, when left to itself, the CNN consistently plants all wheat leading to zero reward.
(ALL-WHEAT FARM)
This is certainly odd behavior as there are a total of 94 all-wheat farm states out of 117018 total states in the training set.
It's possible that our training data doesn't help the model generalize enough with the starting cells since reward is only given at the end. The decent performance during training, is evidence for this as it has a random exploration aiding its crop choice. 

#### Adding in two more crops

Considering it's the network's job to find best possible placement of crops, we thought it'd be interesting to apply it on the 4-crop farm regardless of the fact that it wasn't able to completely solve the 2-crop version. 
It's particularly interesting because the added constraints make it difficult for a random algorithm to reach beyond (INSERT RANDOM HERE). 
Therefore any decisions from the network that lead to that can be considered significant. 

![alt text](https://github.com/Farbod909/cs175-dont-starve/blob/master/figures/4_crop_avg_reward.PNG)
