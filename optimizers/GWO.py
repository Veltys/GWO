# -*- coding: utf-8 -*-


"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""


import random
import time

import numpy

from solution import solution


def GWO(objf, lb, ub, dim, searchAgentsNo, maxIter):

    # maxIter = 1000
    # lb = -100
    # ub = 100
    # dim = 30
    # searchAgentsNo = 5

    # initialize alpha, beta, and delta_pos
    alphaPos = numpy.zeros(dim)
    alphaScore = float("inf")

    betaPos = numpy.zeros(dim)
    betaScore = float("inf")

    deltaPos = numpy.zeros(dim)
    deltaScore = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    positions = numpy.zeros((searchAgentsNo, dim))
    for i in range(dim):
        positions[:, i] = (
            numpy.random.uniform(0, 1, searchAgentsNo) * (ub[i] - lb[i]) + lb[i]
        )

    convergenceCurve = numpy.zeros(maxIter)
    s = solution()

    # Loop counter
    print('GWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, maxIter):
        for i in range(0, searchAgentsNo):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                positions[i, j] = numpy.clip(positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < alphaScore:
                deltaScore = betaScore  # Update delte
                deltaPos = betaPos.copy()
                betaScore = alphaScore  # Update beta
                betaPos = alphaPos.copy()
                alphaScore = fitness
                # Update alpha
                alphaPos = positions[i, :].copy()

            if fitness > alphaScore and fitness < betaScore:
                deltaScore = betaScore  # Update delte
                deltaPos = betaPos.copy()
                betaScore = fitness  # Update beta
                betaPos = positions[i, :].copy()

            if fitness > alphaScore and fitness > betaScore and fitness < deltaScore:
                deltaScore = fitness  # Update delta
                deltaPos = positions[i, :].copy()

        a = 2 - l * ((2) / maxIter)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, searchAgentsNo):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * alphaPos[j] - positions[i, j])
                # Equation (3.5)-part 1
                X1 = alphaPos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * betaPos[j] - positions[i, j])
                # Equation (3.5)-part 2
                X2 = betaPos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * deltaPos[j] - positions[i, j])
                # Equation (3.5)-part 3
                X3 = deltaPos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)

        convergenceCurve[l] = alphaScore

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(alphaScore)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "GWO"
    s.objfname = objf.__name__

    return s
