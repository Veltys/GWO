# -*- coding: utf-8 -*-


"""
Created on Mon May 16 14:19:49 2016

@author: hossam
"""


import math
import random
import time

import numpy

from solution import solution


def WOA(objf, lb, ub, dim, searchAgentsNo, maxIter):

    # dim = 30
    # searchAgentsNo = 50
    # lb = -100
    # ub = 100
    # maxIter = 500
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # initialize position vector and score for the leader
    leaderPos = numpy.zeros(dim)
    leaderScore = float("inf")  # change this to -inf for maximization problems

    # Initialize the positions of search agents
    positions = numpy.zeros((searchAgentsNo, dim))
    for i in range(dim):
        positions[:, i] = (
            numpy.random.uniform(0, 1, searchAgentsNo) * (ub[i] - lb[i]) + lb[i]
        )

    # Initialize convergence
    convergence_curve = numpy.zeros(maxIter)

    ############################
    s = solution()

    print('WOA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < maxIter:
        for i in range(0, searchAgentsNo):

            # Return back the search agents that go beyond the boundaries of the search space

            # positions[i,:]=checkBounds(positions[i,:],lb,ub)
            for j in range(dim):
                positions[i, j] = numpy.clip(positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(positions[i, :])

            # Update the leader
            if fitness < leaderScore:  # Change this to > for maximization problem
                leaderScore = fitness
                # Update alpha
                leaderPos = positions[
                    i, :
                ].copy()  # copy current whale position into the leader position

        a = 2 - t * ((2) / maxIter)
        # a decreases linearly fron 2 to 0 in Eq. (2.3)

        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 + t * ((-1) / maxIter)

        # Update the Position of search agents
        for i in range(0, searchAgentsNo):
            r1 = random.random()  # r1 is a random number in [0,1]
            r2 = random.random()  # r2 is a random number in [0,1]

            A = 2 * a * r1 - a  # Eq. (2.3) in the paper
            C = 2 * r2  # Eq. (2.4) in the paper

            b = 1
            #  parameters in Eq. (2.5)
            l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)

            p = random.random()  # p in Eq. (2.6)

            for j in range(0, dim):

                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(
                            searchAgentsNo * random.random()
                        )
                        X_rand = positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - positions[i, j])
                        positions[i, j] = X_rand[j] - A * D_X_rand

                    elif abs(A) < 1:
                        D_Leader = abs(C * leaderPos[j] - positions[i, j])
                        positions[i, j] = leaderPos[j] - A * D_Leader

                elif p >= 0.5:

                    distance2Leader = abs(leaderPos[j] - positions[i, j])
                    # Eq. (2.5)
                    positions[i, j] = (
                        distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                        + leaderPos[j]
                    )

        convergence_curve[t] = leaderScore
        if t % 1 == 0:
            print(
                ["At iteration " + str(t) + " the best fitness is " + str(leaderScore)]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "WOA"
    s.objfname = objf.__name__
    s.best = leaderScore
    s.bestIndividual = leaderPos

    return s
