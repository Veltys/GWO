# -*- coding: utf-8 -*-


"""
Created on Thu May 26 02:00:55 2016

@author: hossam
"""


import random
import time

import numpy

from solution import solution


def BAT(objf, lb, ub, dim, n, maxIteration):

    n = n
    # Population size

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    nGen = maxIteration  # Number of generations

    a = 0.5
    # Loudness  (constant or decreasing)
    r = 0.5
    # Pulse rate (constant or decreasing)

    qMin = 0  # Frequency minimum
    qMax = 2  # Frequency maximum

    d = dim  # Number of dimensions

    # Initializing arrays
    q = numpy.zeros(n)  # Frequency
    v = numpy.zeros((n, d))  # Velocities
    convergenceCurve = []

    # Initialize the population/solutions
    sol = numpy.zeros((n, d))
    for i in range(dim):
        sol[:, i] = numpy.random.rand(n) * (ub[i] - lb[i]) + lb[i]

    # s = numpy.zeros((n, d))
    s = numpy.copy(sol)
    fitness = numpy.zeros(n)

    # initialize solution for the final results
    s = solution()
    print('BAT is optimizing  "' + objf.__name__ + '"')

    # Initialize timer for the experiment
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%s")

    # Evaluate initial random solutions
    for i in range(0, n):
        fitness[i] = objf(sol[i, :])

    # Find the initial best solution and minimum fitness
    ii = numpy.argmin(fitness)
    best = sol[ii, :]
    fmin = min(fitness)

    # Main loop
    for t in range(0, nGen):

        # Loop over all bats(solutions)
        for i in range(0, n):
            q[i] = qMin + (qMin - qMax) * random.random()
            v[i, :] = v[i, :] + (sol[i, :] - best) * q[i]
            s[i, :] = sol[i, :] + v[i, :]

            # Check boundaries
            for j in range(d):
                sol[i, j] = numpy.clip(sol[i, j], lb[j], ub[j])

            # Pulse rate
            if random.random() > r:
                s[i, :] = best + 0.001 * numpy.random.randn(d)

            # Evaluate new solutions
            fNew = objf(s[i, :])

            # Update if the solution improves
            if (fNew <= fitness[i]) and (random.random() < a):
                sol[i, :] = numpy.copy(s[i, :])
                fitness[i] = fNew

            # Update the current best solution
            if fNew <= fmin:
                best = numpy.copy(s[i, :])
                fmin = fNew

        # update convergence curve
        convergenceCurve.append(fmin)

        if t % 1 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(fmin)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%s")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "BAT"
    s.objfname = objf.__name__

    return s
