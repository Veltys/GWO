# -*- coding: utf-8 -*-


# Sine Cosine OPtimization Algorithm


import random
import time

import numpy

from solution import solution


def SCA(objf, lb, ub, dim, SearchAgents_no, maxIter):

    # destination_pos
    destPos = numpy.zeros(dim)
    destScore = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    convergenceCurve = numpy.zeros(maxIter)
    s = solution()

    # Loop counter
    print('SCA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(0, maxIter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                positions[i, j] = numpy.clip(positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(positions[i, :])

            if fitness < destScore:
                destScore = fitness  # Update Dest_Score
                destPos = positions[i, :].copy()

        # Eq. (3.4)
        a = 2
        maxIteration = maxIter
        r1 = a - l * ((a) / maxIteration)  # r1 decreases linearly from a to 0

        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = (2 * numpy.pi) * random.random()
                r3 = 2 * random.random()
                r4 = random.random()

                # Eq. (3.3)
                if r4 < (0.5):
                    # Eq. (3.1)
                    positions[i, j] = positions[i, j] + (
                        r1 * numpy.sin(r2) * abs(r3 * destPos[j] - positions[i, j])
                    )
                else:
                    # Eq. (3.2)
                    positions[i, j] = positions[i, j] + (
                        r1 * numpy.cos(r2) * abs(r3 * destPos[j] - positions[i, j])
                    )

        convergenceCurve[l] = destScore

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(destScore)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "SCA"
    s.objfname = objf.__name__

    return s
