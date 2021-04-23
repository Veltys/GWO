# -*- coding: utf-8 -*-


"""
Created on Mon May 16 10:42:18 2016

@author: hossam
"""


import math
import random
import time

import numpy

from solution import solution


def MFO(objf, lb, ub, dim, n, maxIteration):

    # maxIteration = 1000
    # lb = -100
    # ub = 100
    # dim = 30
    n = 50  # Number of search agents
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of moths
    mothPos = numpy.zeros((n, dim))
    for i in range(dim):
        mothPos[:, i] = numpy.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]
    mothFitness = numpy.full(n, float("inf"))
    # mothFitness=numpy.fell(float("inf"))

    convergenceCurve = numpy.zeros(maxIteration)

    sortedPopulation = numpy.copy(mothPos)
    fitnessSorted = numpy.zeros(n)
    #####################
    bestFlames = numpy.copy(mothPos)
    bestFlameFitness = numpy.zeros(n)
    ####################
    doublePopulation = numpy.zeros((2 * n, dim))
    doubleFitness = numpy.zeros(2 * n)

    doubleSortedPopulation = numpy.zeros((2 * n, dim))
    doubleFitnessSorted = numpy.zeros(2 * n)
    #########################
    previousPopulation = numpy.zeros((n, dim))
    previousFitness = numpy.zeros(n)

    s = solution()

    print('MFO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    iteration = 1

    # Main loop
    while iteration < maxIteration:

        # Number of flames Eq. (3.14) in the paper
        flameNo = round(n - iteration * ((n - 1) / maxIteration))

        for i in range(0, n):

            # Check if moths go out of the search spaceand bring it back
            for j in range(dim):
                mothPos[i, j] = numpy.clip(mothPos[i, j], lb[j], ub[j])

            # evaluate moths
            mothFitness[i] = objf(mothPos[i, :])

        if iteration == 1:
            # Sort the first population of moths
            fitnessSorted = numpy.sort(mothFitness)
            ii = numpy.argsort(mothFitness)

            sortedPopulation = mothPos[ii, :]

            # Update the flames
            bestFlames = sortedPopulation
            bestFlameFitness = fitnessSorted
        else:
            #
            #        # Sort the moths
            doublePopulation = numpy.concatenate(
                (previousPopulation, bestFlames), axis=0
            )
            doubleFitness = numpy.concatenate(
                (previousFitness, bestFlameFitness), axis=0
            )
            #
            doubleFitnessSorted = numpy.sort(doubleFitness)
            i2 = numpy.argsort(doubleFitness)
            #
            #
            for newindex in range(0, 2 * n):
                doubleSortedPopulation[newindex, :] = numpy.array(
                    doublePopulation[i2[newindex], :]
                )

            fitnessSorted = doubleFitnessSorted[0:n]
            sortedPopulation = doubleSortedPopulation[0:n, :]
            #
            #        # Update the flames
            bestFlames = sortedPopulation
            bestFlameFitness = fitnessSorted

        #
        #   # Update the position best flame obtained so far
        bestFlameScore = fitnessSorted[0]
        # Best_flame_pos = sortedPopulation[0, :]
        #
        previousPopulation = mothPos
        previousFitness = mothFitness
        #
        # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + iteration * ((-1) / maxIteration)

        # Loop counter
        for i in range(0, n):
            #
            for j in range(0, dim):
                if (
                    i <= flameNo
                ):  # Update the position of the moth with respect to its corresponsing flame
                    #
                    # D in Eq. (3.13)
                    distanceToFlame = abs(sortedPopulation[i, j] - mothPos[i, j])
                    b = 1
                    t = (a - 1) * random.random() + 1
                    #
                    #                % Eq. (3.12)
                    mothPos[i, j] = (
                        distanceToFlame * math.exp(b * t) * math.cos(t * 2 * math.pi)
                        + sortedPopulation[i, j]
                    )
                #            end
                #
                if (
                    i > flameNo
                ):  # Upaate the position of the moth with respct to one flame
                    #
                    #                % Eq. (3.13)
                    distanceToFlame = abs(sortedPopulation[i, j] - mothPos[i, j])
                    b = 1
                    t = (a - 1) * random.random() + 1
                    #
                    #                % Eq. (3.12)
                    mothPos[i, j] = (
                        distanceToFlame * math.exp(b * t) * math.cos(t * 2 * math.pi)
                        + sortedPopulation[flameNo, j]
                    )

        convergenceCurve[iteration] = bestFlameScore
        # Display best fitness along the iteration
        if iteration % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(iteration)
                    + " the best fitness is "
                    + str(bestFlameScore)
                ]
            )

        iteration = iteration + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "MFO"
    s.objfname = objf.__name__

    return s
