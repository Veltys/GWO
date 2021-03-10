# -*- coding: utf-8 -*-


""" JAYA Algorithm """

import random
import time

import numpy

from solution import solution


def JAYA(objf, lb, ub, dim, searchAgentsNo, maxIter):

    # Best and Worst position initialization
    bestPos = numpy.zeros(dim)
    bestScore = float("inf")

    worstPos = numpy.zeros(dim)
    worstScore = float(0)

    fitnessMatrix = numpy.zeros((searchAgentsNo))

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

    for i in range(0, searchAgentsNo):

        # Return back the search agents that go beyond the boundaries of the search space
        for j in range(dim):
            positions[i, j] = numpy.clip(positions[i, j], lb[j], ub[j])

        # Calculate objective function for each search agent
        fitness = objf(positions[i])
        fitnessMatrix[i] = fitness

        if fitness < bestScore:
            bestScore = fitness  # Update Best_Score
            bestPos = positions[i]

        if fitness > worstScore:
            worstScore = fitness  # Update Worst_Score
            worstPos = positions[i]

    convergenceCurve = numpy.zeros(maxIter)
    s = solution()

    # Loop counter
    print('JAYA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(0, maxIter):

        # Update the Position of search agents
        for i in range(0, searchAgentsNo):
            newPosition = numpy.zeros(dim)
            for j in range(0, dim):

                # Update r1, r2
                r1 = random.random()
                r2 = random.random()

                # JAYA Equation
                newPosition[j] = (
                    positions[i][j]
                    + r1 * (bestPos[j] - abs(positions[i, j]))
                    - r2 * (worstPos[j] - abs(positions[i, j]))
                )

                # checking if newPosition[j] lies in search space
                if newPosition[j] > ub[j]:
                    newPosition[j] = ub[j]
                if newPosition[j] < lb[j]:
                    newPosition[j] = lb[j]

            newFitness = objf(newPosition)
            currentFit = fitnessMatrix[i]

            # replacing current element with new element if it has better fitness
            if newFitness < currentFit:
                positions[i] = newPosition
                fitnessMatrix[i] = newFitness

        # finding the best and worst element
        for i in range(searchAgentsNo):
            if fitnessMatrix[i] < bestScore:
                bestScore = fitnessMatrix[i]
                bestPos = positions[i, :].copy()

            if fitnessMatrix[i] > worstScore:
                worstScore = fitnessMatrix[i]
                worstPos = positions[i, :].copy()

        convergenceCurve[l] = bestScore

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(bestScore)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "JAYA"
    s.objfname = objf.__name__

    return s
