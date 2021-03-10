# -*- coding: utf-8 -*-


import math
import random
import time

import numpy

from solution import solution


def SSA(objf, lb, ub, dim, n, maxIteration):

    # maxIteration = 1000
    # lb = -100
    # ub = 100
    # dim = 30
    n = 50  # Number of search agents
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    convergenceCurve = numpy.zeros(maxIteration)

    # Initialize the positions of salps
    SalpPositions = numpy.zeros((n, dim))
    for i in range(dim):
        SalpPositions[:, i] = numpy.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]
    SalpFitness = numpy.full(n, float("inf"))

    # FoodPosition = numpy.zeros(dim)
    # FoodFitness = float("inf")
    # Moth_fitness=numpy.fell(float("inf"))

    s = solution()

    print('SSA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for i in range(0, n):
        # evaluate moths
        SalpFitness[i] = objf(SalpPositions[i, :])

    sorted_salps_fitness = numpy.sort(SalpFitness)
    I = numpy.argsort(SalpFitness)

    Sorted_salps = numpy.copy(SalpPositions[I, :])

    FoodPosition = numpy.copy(Sorted_salps[0, :])
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 1

    # Main loop
    while Iteration < maxIteration:

        # Number of flames Eq. (3.14) in the paper
        # Flame_no=round(n-Iteration*((n-1)/maxIteration));

        c1 = 2 * math.exp(-((4 * Iteration / maxIteration) ** 2))
        # Eq. (3.2) in the paper

        for i in range(0, n):

            SalpPositions = numpy.transpose(SalpPositions)

            if i < n / 2:
                for j in range(0, dim):
                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )

                    ####################

            elif i >= n / 2 and i < n + 1:
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]

                SalpPositions[:, i] = (point2 + point1) / 2
                # Eq. (3.4) in the paper

            SalpPositions = numpy.transpose(SalpPositions)

        for i in range(0, n):

            # Check if salps go out of the search spaceand bring it back
            for j in range(dim):
                SalpPositions[i, j] = numpy.clip(SalpPositions[i, j], lb[j], ub[j])

            SalpFitness[i] = objf(SalpPositions[i, :])

            if SalpFitness[i] < FoodFitness:
                FoodPosition = numpy.copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]

        # Display best fitness along the iteration
        if Iteration % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(Iteration)
                    + " the best fitness is "
                    + str(FoodFitness)
                ]
            )

        convergenceCurve[Iteration] = FoodFitness

        Iteration = Iteration + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "SSA"
    s.objfname = objf.__name__

    return s
