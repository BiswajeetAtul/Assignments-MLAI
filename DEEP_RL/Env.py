# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
no_of_locations = 5 # number of cities, ranges from 1 ..... m
no_of_hours = 24 # number of hours, ranges from 0 .... t-1
no_of_days = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
##############
class items(Enum):
    loc="loc"
    time="time"
    day="day"
    pickup="pickup"
    drop="drop"

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # (0,0) signifies that the cab driver goes offline.
        # permutations function gives  (𝑚 − 1) ∗ 𝑚 + 1 items for m locations.
        self.action_space = [(0,0)]+list(permutations([i for i in range(no_of_locations)],2))
        # state space is a tuple of combinations of location,hours,days
        self.state_space = [(x,y,z) for x in range(no_of_locations) for y in range(no_of_hours) for z in range(no_of_days)]
        # the initial state is randomly selected.
        self.state_init = random.choice(self.state_space)
        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
    #    """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encode=[0]*(no_of_locations+no_of_hours+no_of_days)    
        state_encode[self.get_set_state_action(state,items.loc)]=1
        state_encode[no_of_locations+ self.get_set_state_action(state,items.time)]=1
        state_encode[no_of_locations+no_of_hours+ lf.get_set_state_action(state,items.day)]=1
        return state_encod


    # Use this function if you are using architecture-2 
    #def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_idx = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_idx,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init

    def get_set_state_action(self, item_list, get_item=None, set_item=None,set_item_val=None):
        if(get_item==items.loc or get_item==items.pickup):
            return item_list[0]
        elif(get_item==items.time or get_item==items.drop):
            return item_list[1]
        elif(get_item==items.day):
            return item_list[2]
        elif(set_item==items.loc or set_item==items.pickup):
            item_list[0]=set_item_val
        elif(set_item==items.time or set_items==items.drop):
            item_list[1]=set_item_val
        elif(set_item==items.day):
            item_list[2]=set_item_val

