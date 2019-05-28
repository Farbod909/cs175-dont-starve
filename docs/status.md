---
layout: default
title: Status
---

## Project Summary

While our main goal is still to create an agent that can learn to farm, we have decided to change the agent's reason for farming. Initially, we had the idea that we could make an agent motivated by in-game hunger, but that turned out not to be an interesting challenge, since hunger is very slow to deplete and growing enough food to survive is fairly easy.

Our new idea is to require (with a mod) that crops be planted in certain configurations in order to grow at all. Our agent would then have to learn how to plant these crops to make efficient use of its limited farming space.

The environment is deterministic (a square farm of a given size), so the agent does not need any percepts while planting, only to remember what crops it has planted and where. This also makes the agent very fast, since waiting for Malmo's percepts such as a grid view is slow and often unreliable. After the agent is done planting, it harvests the crops and counts them to find its reward for that iteration.

## Approach

## Evaluation

## Video

## Remaining Goals and Challenges

## Resources Used
