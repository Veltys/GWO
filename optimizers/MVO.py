# -*- coding: utf-8 -*-


"""
Created on Wed May 11 17:06:34 2016

@author: hossam
"""


import math
import random

from numpy import asarray
import numpy

from sklearn.preprocessing import normalize
from solution import solution


def normr(mat):
    """normalize the columns of the matrix
    B= normr(A) normalizes the row
    the dtype of A is float"""
    mat = mat.reshape(1, -1)
    # Enforce dtype float
    if mat.dtype != "float":
        mat = asarray(mat, dtype=float)

    # if statement to enforce dtype float
    B = normalize(mat, norm="l2", axis=1)
    B = numpy.reshape(B, -1)
    return B


def randk(t):
    if (t % 2) == 0:
        s = 0.25
    else:
        s = 0.75
    return s


def rouletteWheelSelection(weights):
    accumulations = numpy.cumsum(weights)
    p = random.random() * accumulations[-1]
    chosen_index = -1
    for index, accumulation in enumerate(accumulations):
        if accumulation > p:
            chosen_index = index
            break

    choice = chosen_index

    return choice


def MVO(objf, lb, ub, dim, n, maxTime):

    # parameters
    # dim = 30
    # lb = -100
    # ub = 100
    wepMax = 1
    wepMin = 0.2
    # maxTime=1000
    # n=50
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    universes = numpy.zeros((n, dim))
    for i in range(dim):
        universes[:, i] = numpy.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]

    Sorted_universes = numpy.copy(universes)

    convergence = numpy.zeros(maxTime)

    bestUniverse = [0] * dim
    bestUniverseInflationRate = float("inf")

    s = solution()

    time = 1
    ############################################
    print('MVO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime('%Y-%m-%d-%H-%M-%S')
    while time < maxTime + 1:

        # Eq. (3.3) in the paper
        wep = wepMin + time * ((wepMax - wepMin) / maxTime)

        tdr = 1 - (math.pow(time, 1 / 6) / math.pow(maxTime, 1 / 6))

        inflationRates = [0] * len(universes)

        for i in range(0, n):
            for j in range(dim):
                universes[i, j] = numpy.clip(universes[i, j], lb[j], ub[j])

            inflationRates[i] = objf(universes[i, :])

            if inflationRates[i] < bestUniverseInflationRate:

                bestUniverseInflationRate = inflationRates[i]
                bestUniverse = numpy.array(universes[i, :])

        sortedInflationRates = numpy.sort(inflationRates)
        sortedIndexes = numpy.argsort(inflationRates)

        for newindex in range(0, n):
            Sorted_universes[newindex, :] = numpy.array(
                universes[sortedIndexes[newindex], :]
            )

        normalizedSortedInflationRates = numpy.copy(normr(sortedInflationRates))

        universes[0, :] = numpy.array(Sorted_universes[0, :])

        for i in range(1, n):
            backHoleIndex = i
            for j in range(0, dim):
                r1 = random.random()

                if r1 < normalizedSortedInflationRates[i]:
                    whiteHoleIndex = rouletteWheelSelection(-sortedInflationRates)

                    if whiteHoleIndex == -1:
                        whiteHoleIndex = 0
                    whiteHoleIndex = 0
                    universes[backHoleIndex, j] = Sorted_universes[
                        whiteHoleIndex, j
                    ]

                r2 = random.random()

                if r2 < wep:
                    r3 = random.random()
                    if r3 < 0.5:
                        universes[i, j] = bestUniverse[j] + tdr * (
                            (ub[j] - lb[j]) * random.random() + lb[j]
                        )  # random.uniform(0,1)+lb);
                    if r3 > 0.5:
                        universes[i, j] = bestUniverse[j] - tdr * (
                            (ub[j] - lb[j]) * random.random() + lb[j]
                        )  # random.uniform(0,1)+lb);

        convergence[time - 1] = bestUniverseInflationRate
        if time % 1 == 0:
            print(
                [
                    'At iteration '
                    + str(time)
                    +' the best fitness is '
                    + str(bestUniverseInflationRate)
                ]
            )

        time = time + 1
    timerEnd = time.time()
    s.endTime = time.strftime('%Y-%m-%d-%H-%M-%S')
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = 'MVO'
    s.objfname = objf.__name__

    return s
