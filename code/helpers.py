import numpy as np


def state_as_one_hot(state): 
	if isinstance(state, tuple):
		state = state[0]
	one_hot = np.array([])
	for cell in state:
		one_hot = np.concatenate([one_hot, np.identity(4)[cell+1]])
	return one_hot