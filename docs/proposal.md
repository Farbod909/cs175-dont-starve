---
layout: default
title:  Proposal
---
## Summary of the Project
We are looking to create an agent that can learn to farm. Our initial state will be flat grasslands and the goal is for the agent to not let itself starve to death by planting crops and feeding itself. The agent will explore multiple types of crops and determine the best crops to feed itself with.

Input: Grid of blocks surrounding the agent
Output: Action (i.e. move left, right click, etc)

## AI/ML Algorithms
Reinforcement learning with Markov models

## Evaluation Plan
Initial metric is how long the agent lasts before it starves, with the baseline being however much time it takes for an idle agent to starve. Eventually it won’t starve because it learned to grow enough crops to survive. Therefore our second metric is how much surplus crops it grows. Lastly, the agent will be given a time limit of 10x the time it takes to starve.

Our sanity check is if the agent can plant and harvest at least one crop. One way to visualize the internals of the algorithms is to show that it will avoid to stay idle at all costs as standing around puts the agent at risk of starving. To help the agent learn we will give it some reward for intermediate steps in the farming process, such as tilling ground and planting seeds. 

The best case scenario we anticipate is if our agent can feed itself and also produce a surplus. It’d interesting to see if certain parameters affect which crops the agent plants and if it organizes them in a clean manner. We imagine that initially it will plant crops anywhere, but it’ll be interesting if it learns that planting crops in a compact manner saves it time. 
