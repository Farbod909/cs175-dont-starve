## Project Summary
Our environment is a simple grid farm that is ready for planting.
Our agent's task is to determine which plants to crop at each particular cell of the farm.
The challenge, however, is that crops are required to be planted in certain configurations in order to grow at all. 
Our agent would then have to learn how to plant these crops to make efficient use of its limited farming space.

The required configurations are as follows:
  - Wheat needs to be planted in 1-wide rows or columns, with at least 2 wheat per line.
  - Potatoes need to be planted with two other potatoes nearby, in an L shape, or with three other potatoes.
  - Carrots will only grow if they have zero or one nearby carrots.
  - Beetroots require four or more adjacent beetroots, including diagonals.

With a small farm such as a 3x3 farm with only 2 crops in the agent's inventory, it is possible to apply tabular Q-learning as there are a total of 2^9 (512) possible states (2 crops per cell in the farm). 
The tabular method, however, is not applicable when using all the crops as that gives us 4^9 (262,144). 
Moreover, increasing the height and width of the farm by just one increases our possible states to 4^16 (4,294,967,296) possible states which is clearly infeasible with the tabular method.

## Approaches
Considering the large number of states, we believe Deep Q-Learning is necessary to solve our task. 
In our implementation we attempt to solve the small 2^9 problem described above using a feed-forward neural network as well as a convolutional neural network before taking on the 4^9 farm. 
Particularly, for each network, we iteratively update the Q-values by sampling from replay memory at the end of every episode. 

### Baseline
Our main metric of performance is average reward per episode.
We can obtain a baseline of performance by observing how an algorithm that plants crops randomly performs under the constraints we set.


![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/avg_random_reward.PNG)


### Feed Forward Neural Network
In our previous report, we trained a feed-foward neural network with 5 hidden layers & 15 neurons per layer on our farm over 1000 iterations. 
We did not find that it was an effective approach, but decided to make the network bigger and run it over 10,000 iterations to obtain stronger evidence that this type of network can't fit our problem set. 
The average reward during training for the larger network is shown in the figure below:


![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/r-list-10k-avg.png)


Regardless of the fact that we gave it 10x more data and increased the number of layers and nodes, a feed forward neural network does not seem to be complex enough to fit our problem set. 
As training progresses, it begins to do worse than random.

### Convolutional Neural Network (CNN)
CNNs are best known for their high performance in machine vision as the filters in the convolution layers aid it in recognizing patterns amongst the pixels. 
Considering our task is to find plant formations that lead to maximum crop growth, a CNN has potential to fit the problem set if we translate the farm into a format that's similar to single-channel images.
We do this by assigning every crop an ID and passing in our farm as a 3d matrix of the shape [1,3,3].
This matrix is then passed to our CNN that holds 2 convolutional layers and 3 linear layers.


![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/Example%20Input.PNG)

![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/cropped_cnn_fig.png)


We trained our network using greedy-epsilon with epsilon set to .5. Plotting our average reward as we train, we find that the convolutional neural network does significantly better than the feed-forward neural network. 
As we reach 10,000 episodes, the average reward starts reaching that of the random baseline. 
It is, however, difficult to tell if it will do better than random and can only be proved by training the network over more episodes.


![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/2_crop_avg_reward.PNG)


To test our CNN, we load the trained model and set exploration to 0.
Unfortunately, when left to itself, the CNN consistently plants all wheat leading to zero reward.


![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/2_crop_dec_test_avg_reward.PNG)


This is certainly odd behavior as there are a total of 94 all-wheat farm states out of 117,018 total states in the training set.
It's possible that our training data doesn't help the model generalize enough with the starting cells since reward is only given at the end. The decent performance during training, is evidence for this as it has a random exploration aiding its crop choice. 

#### Adding in two more crops
We thought it'd be worthwhile to apply our CNN on the 4-crop farm regardless of the fact that it wasn't able to completely solve the 2-crop version. 
We find it worthwhile because the added constraints make it more difficult to achieve high reward but the new crops allow for many more configurations for the network to try. 
Therefore, any consistent decisions from the network that lead to high reward can be considered significant. 
The following chart shows its performance over the training phase:


![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/4_crop_avg_reward.PNG) \\


For the testing phase, we set a decreasing exploration rate from .3 to 0 over the first 200 episodes.
This network managed to get an average reward rate 3.5, which is a very stark from the 2-crop CNN's behavior.
![alt text](https://raw.githubusercontent.com/Farbod909/cs175-dont-starve/master/figures/4_crop_dec_test_avg_reward.PNG)
Further delving into this network's behaviour we find that it seems to cycle through many different strategies, including ones with high reward, 0 reward, or middling reward. 
For example, every 20 or 30 episodes it will plant an all wheat farm and other times it will plant configurations that lead to high reward.

## Summary & Conclusion 
As was seen above, it seems that a CNN may be capable of fitting our problem set if given enough training time and possibly more layers. 
During training, it was able to reach a decent amount of reward according to our baseline which is probably due to its filters which are known to recognize patterns. 
It is odd though that, during the testing phase, the 2-crop CNN was unable to plant anything beyond the all-wheat farm while the 4-crop CNN managed to place crops in a variety of configurations. 
There are many potential explanations for this beyond what we discussed above such as the significant increase of possible configurations from 2 crops to 4 crops. 
Unfortunately, due to time constraints, we were not able to inspect the 2-crop CNN beyond checking if there were all-wheat farms in the training set. 
It's behavior did inspire us to try placing an exploration rate during the start of the test-phase for the 4-crop, so another possible inspection is to do the same for the 2-crop (though we doubt it will cause a change in behavior). 


Other potential options is to provide negative rewards at the end if no crops grow. This way there is a strong differentiator between states where the agent is still cropping and states where plants failed to grow at all.

## Video
If you'd like to see the networks in action, feel free to view the video below:
<iframe width="560" height="315" src="https://www.youtube.com/embed/vg_Oxh8irr8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
