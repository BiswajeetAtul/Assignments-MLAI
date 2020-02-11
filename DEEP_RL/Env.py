# Import routines
import numpy as np
import sys
import math
import random
from itertools import permutations
from enum import Enum

# Defining hyperparameters
no_of_locations = 5 # number of cities, ranges from 1 .....  m
no_of_hours = 24 # number of hours, ranges from 0 ....  t-1
no_of_days = 7  # number of days, ranges from 0 ...  d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
##############
class items(Enum):
    loc = "loc"
    time = "time"
    day = "day"
    pickup = "pickup"
    drop = "drop"

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # (0,0) signifies that the cab driver goes offline.
        # permutations function gives (𝑚 − 1) ∗ 𝑚 + 1 items for m locations.
        self.action_space = [(0,0)] + list(permutations([i for i in range(no_of_locations)],2))
        # state space is a tuple of combinations of location,hours,days
        self.state_space = [(x,y,z) for x in range(no_of_locations) for y in range(no_of_hours) for z in range(no_of_days)]
        # the initial state is randomly selected.
        self.state_init = random.choice(self.state_space)
        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def stateEncodeArc1(self, state):
    #    """convert the state into a vector so that it can be fed to the NN.
    #    This method converts a given state into a vector format.  Hint: The
    #    vector is of size m + t + d."""
        state_encode = [0] * (no_of_locations + no_of_hours + no_of_days)    
        state_encode[self.get_set_state_action(state,items.loc)] = 1
        state_encode[no_of_locations + self.get_set_state_action(state,items.time)] = 1
        state_encode[no_of_locations + no_of_hours + lf.get_set_state_action(state,items.day)] = 1
        return state_encod


    # Use this function if you are using architecture-2
    #def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to
    #     the NN.  This method converts a given state-action pair into a vector
    #     format.  Hint: The vector is of size m + t + d + m + m."""

        
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

        if requests > 15:
            requests = 15

        possible_actions_idx = random.sample(range(1, (m - 1) * m + 1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_idx,actions   



    def rewardFunc(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        """
            Objective: Maximize the reward of the Driver.
            If C is the amount of battery consumed per hour and R, the revenue of the from the ride, for every hour, then
            Reward function is defined as: R(state=XiTjDk)= 
                            (revenue earned from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) - 
                            (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝)
        """
        ride_time = None
        idle_time = None
        reward = (R * ride_time) - (C * (ride_time + idle_time))

        return reward

    def updateTimeDay(self, time, day, ride_duration):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        ride_duration = int(ride_duration)

        if (time + ride_duration) < 24:
            time = time + ride_duration
            # day is unchanged
        else:
            # duration taken spreads over to subsequent days
            # convert the time to 0-23 range
            time = (time + ride_duration) % 24 
            
            # Get the number of days
            num_days = (time + ride_duration) // 24
            
            # Convert the day to 0-6 range
            day = (day + num_days) % 7

        return time, day


    def nextStateFunc(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []

        #Time taken from pickup point 𝑝 to drop point 𝑞
        time_for_ride = 0
        #Time taken in moving from current point 𝑖 to pick-up point 𝑝
        time_for_transit = 0
        #Time taken if the driver refuses the request
        time_for_idle_refusal = 0


        # Now from the state and action we will get the current locations, day
        # and time.
        # these will be used as index from the Time Matrix
        current_location = self.getSetStateAction(state,items.loc)
        pickup_location = self.getSetStateAction(action,items.pickup)
        drop_location = self.getSetStateAction(action,items.drop)
        current_time = self.getSetStateAction(state,items.time)
        current_day = self.getSetStateAction(state, items.day)

        # next location stores the next location after the action taken
        next_location = drop_location
        # The state transition can be two, depending on the current state, and
        # the action choosen.
        #if action == (0,0): the driver refuses the ride.  the locations will
        #remain same, but time will increase.
        #if action == (p,q): the driver accepts the ride.  the locations as
        #well as the time will change.
        #if action == (p,q) and the driver is already at p, then the driver
        #accepts the ride.  the locations as well as the time will change, but
        #the trasition time will be 0.

        if(pickup_location == 0 and drop_location == 0):
            # the time increases
            time_for_idle_refusal = 1
            
        elif(pickup_location == current_location):
            # the driver is alreacy at p
            time_for_idle_refusal = 0
            time_for_transit = 0
            time_for_ride = Time_matrix[current_location][drop_location][current_time][current_day]
        else:
            time_for_transit = Time_matrix[current_location][pickup_location][current_time][current_day]
            new_date,new_day = self.updateTimeDay(current_time,current_day,time_for_transit)
            time_for_ride = Time_matrix[current_location][drop_location][current_time][current_day]
            

        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init

    def getSetStateAction(self, item_list, get_item=None, set_item=None,set_item_val=None):
        if(get_item == items.loc or get_item == items.pickup):
            return item_list[0]
        elif(get_item == items.time or get_item == items.drop):
            return item_list[1]
        elif(get_item == items.day):
            return item_list[2]
        elif(set_item == items.loc or set_item == items.pickup):
            item_list[0] = set_item_val
        elif(set_item == items.time or set_items == items.drop):
            item_list[1] = set_item_val
        elif(set_item == items.day):
            item_list[2] = set_item_val


def main():
    cdObj = CabDriver()
    print("Done?")
    time_matrix = np.load("TM.npy")
    cdObj.nextStateFunc(cdObj.state_init,(0,0),time_matrix)
    print("Done?")

if __name__ == "__main__":
    sys.exit(int(main() or 0))