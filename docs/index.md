---
layout: default
title:  Home
---

[Source Code Repository](https://github.com/Farbod909/cs175-dont-starve)

<img src="https://i.imgur.com/iqAfsK6.png" width="960">

Dont-Starve is a project to create an agent which can learn how to plant and grow crops efficiently in Minecraft, using Malmo. In vanilla Minecraft crops are very easy to grow, so to make it more interesting we have a mod which prevents crops from growing unless they are planted in a certain way:

Wheat needs to be planted in 1-wide rows or columns, with at least 2 wheat per line.  
Potatoes need to be planted with two other potatoes nearby, in an L shape, or with three other potatoes.  
Carrots will only grow if they have zero or one nearby carrots.  
Beetroots require four or more adjacent beetroots, including diagonals.

The agent will learn how to plant crops in these configurations, to obtain better and better harvests in its limited space, without being taught these rules directly.
