import numpy as np
import pandas as pd
import random as rand

def probability(theta, b):
    p = 1 / (1 + np.exp(-2 * (theta - b)))
    return p

class Elo:
    def elo_probability(b, theta, q): 

        elo_Prob = np.ones(len(theta))
        for i in range(len(theta)):
            if q[i] == 0:
                continue
                
            elo_Prob[i] = probability(theta[i], b) # Compute the probability
        if np.prod(elo_Prob) > rand.random():
            return 1, elo_Prob # right
        else:
            return 0, elo_Prob # wrong 
        
    # Skill level update
    @staticmethod
    def skill_update(b, theta, q, d, k_success, k_fail):
        """
        Updates the skill level based on the outcome of an interaction.

        Parameters:
        - b (float): Question difficulty parameter.
        - theta (array): Skill levels of the user.
        - q (array): A-matrix, contains information about the skill involvement.
        - d (array): Skill dependencies (negative values allow updates, positive restrict updates).
        - k_success (float): Learning rate for correct responses.
        - k_fail (float): Learning rate for incorrect responses.

        Returns:
        - theta (array): Updated skill levels.
        - S (int): Whether the response was correct (1) or incorrect (0).
        """
        S, elo_Prob = Elo.elo_probability(b, theta, q)
        k = k_success if S == 1 else k_fail

        theta = np.copy(theta) # Avoid modifying the original array
        for i in range(len(theta)):
            if q[i] == 0:
                continue
            if d[i] >= 0 and theta[d[i]] < 1:
                continue
            # Update skill level:
            theta[i] = theta[i] + k*(1 - elo_Prob[i])
        return theta, S
