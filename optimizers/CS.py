# -*- coding: utf-8 -*-


"""
Created on Tue May 24 13:13:28 2016

@author: Hossam Faris
"""


import math
import random
import time

import numpy

from solution import solution


def getCuckoos(nest, best, lb, ub, n, dim):

    # perform Levy flights
    # tempnest = numpy.zeros((n, dim))
    tempnest = numpy.array(nest)
    beta = 3 / 2
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    s = numpy.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = numpy.random.randn(len(s)) * sigma
        v = numpy.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * numpy.random.randn(len(s))

        for k in range(dim):
            tempnest[j, k] = numpy.clip(s[k], lb[k], ub[k])

    return tempnest


def getBestNest(nest, newnest, fitness, n, dim, objf):
    # Evaluating all new solutions
    # tempnest = numpy.zeros((n, dim))
    tempnest = numpy.copy(nest)

    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = objf(newnest[j, :])
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmin = min(fitness)
    K = numpy.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness


# Replace some nests by constructing new solutions/nests
def emptyNests(nest, pa, n, dim):

    # Discovered or not
    # tempnest = numpy.zeros((n, dim))

    K = numpy.random.uniform(0, 1, (n, dim)) > pa

    stepsize = random.random() * (
        nest[numpy.random.permutation(n), :] - nest[numpy.random.permutation(n), :]
    )

    tempnest = nest + stepsize * K

    return tempnest


##########################################################################


def CS(objf, lb, ub, dim, n, N_IterTotal):

    # lb=-1
    # ub=1
    # n=50
    # N_IterTotal=1000
    # dim=30

    # Discovery rate of alien eggs/solutions
    pa = 0.25

    #    Lb=[lb]*nd
    #    Ub=[ub]*nd
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # RInitialize nests randomely
    nest = numpy.zeros((n, dim))
    for i in range(dim):
        nest[:, i] = numpy.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]

    # newNest = numpy.zeros((n, dim))
    newNest = numpy.copy(nest)

    # bestnest = [0] * dim

    fitness = numpy.zeros(n)
    fitness.fill(float("inf"))

    s = solution()

    print('CS is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    fmin, bestnest, nest, fitness = getBestNest(nest, newNest, fitness, n, dim, objf)
    convergence = []
    # Main loop counter
    for iterator in range(0, N_IterTotal):
        # Generate new solutions (but keep the current best)

        newNest = getCuckoos(nest, bestnest, lb, ub, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = getBestNest(nest, newNest, fitness, n, dim, objf)

        newNest = emptyNests(newNest, pa, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = getBestNest(nest, newNest, fitness, n, dim, objf)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

        if iterator % 10 == 0:
            print(["At iteration " + str(iterator) + " the best fitness is " + str(fmin)])
        convergence.append(fmin)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "CS"
    s.objfname = objf.__name__

    return s
