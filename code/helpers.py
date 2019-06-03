import numpy as np


def state_as_one_hot(state): 
	one_hot = np.array([])
	for cell in state:
		one_hot = np.concatenate([one_hot, np.identity(6)[cell+1]])