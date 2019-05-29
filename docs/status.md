---
layout: default
title: Status
---

## Project Summary

While our main goal is still to create an agent that can learn to farm, we have decided to change the agent's reason for farming. Initially, we had the idea that we could make an agent motivated by in-game hunger, but that turned out not to be an interesting challenge, since hunger is very slow to deplete and growing enough food to survive is fairly easy.

Our new idea is to require (with a mod) that crops be planted in certain configurations in order to grow at all. Our agent would then have to learn how to plant these crops to make efficient use of its limited farming space.

The environment is deterministic (a square farm of a given size), so the agent does not need any percepts while planting, only to remember what crops it has planted and where. This also makes the agent very fast, since waiting for Malmo's percepts such as a grid view is slow and often unreliable. After the agent is done planting, it harvests the crops and counts them to find its reward for that iteration.

## Approach
At the moment we are using Deep Q-Learning with two different types of neural networks. We are doing a feed-forward neural network and a convolutional neural network to approximate our q-values. 

MDP:
Our MDP is one in which every node is a cell in the farm and every outgoing edge is a potential crop it can plant. Reward is only given in the final state when every cell is full and we count the amount of crops that grew. Though reward seems scarce, we believe it is okay because the agent will always reach the terminating state as it moves throughout the farm deterministically.

## Evaluation
In our project home page we described the following: 

Wheat needs to be planted in 1-wide rows or columns, with at least 2 wheat per line.
Potatoes need to be planted with two other potatoes nearby, in an L shape, or with three other potatoes.
Carrots will only grow if they have zero or one nearby carrots.
Beetroots require four or more adjacent beetroots, including diagonals.

Qualitatively, a well performing model is one that plants according to at least one of the patterns described above. For example, if it plants wheat in rows, then we know it is learning successfully. If it continue to act random, on the other hand, our model is not learning properly.

## Video

## Remaining Goals and Challenges

## Resources Used
http://deeplizard.com/learn/video/0bt0SjbS3xc

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training

https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

