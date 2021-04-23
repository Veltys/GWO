# -*- coding: utf-8 -*-


"""
Created on Thirsday March 21  2019

@author:
% _____________________________________________________
% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems,
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% _____________________________________________________

"""


import math
import random
import time

import numpy

from solution import solution


def HHO(objf, lb, ub, dim, searchAgentsNo, maxIter):

    # dim = 30
    # searchAgentsNo = 50
    # lb = -100
    # ub = 100
    # maxIter = 500

    # initialize the location and Energy of the rabbit
    rabbitLocation = numpy.zeros(dim)
    rabbitEnergy = float("inf")  # change this to -inf for maximization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)

    # Initialize the locations of Harris' hawks
    x = numpy.asarray(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (searchAgentsNo, dim))]
    )

    # Initialize convergence
    convergenceCurve = numpy.zeros(maxIter)

    ############################
    s = solution()

    print('HHO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < maxIter:
        for i in range(0, searchAgentsNo):

            # Check boundries

            x[i, :] = numpy.clip(x[i, :], lb, ub)

            # fitness of locations
            fitness = objf(x[i, :])

            # Update the location of Rabbit
            if fitness < rabbitEnergy:  # Change this to > for maximization problem
                rabbitEnergy = fitness
                rabbitLocation = x[i, :].copy()

        e1 = 2 * (1 - (t / maxIter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, searchAgentsNo):

            e0 = 2 * random.random() - 1  # -1<e0<1
            escapingEnergy = e1 * (
                e0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(escapingEnergy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(searchAgentsNo * random.random())
                xRand = x[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    x[i, :] = xRand - random.random() * abs(
                        xRand - 2 * random.random() * x[i, :]
                    )

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    x[i, :] = (rabbitLocation - x.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            # -------- Exploitation phase -------------------
            elif abs(escapingEnergy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probablity of each event

                if (
                    r >= 0.5 and abs(escapingEnergy) < 0.5
                ):  # Hard besiege Eq. (6) in paper
                    x[i, :] = (rabbitLocation) - escapingEnergy * abs(
                        rabbitLocation - x[i, :]
                    )

                if (
                    r >= 0.5 and abs(escapingEnergy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    jumpStrength = 2 * (
                        1 - random.random()
                    )  # random jump strength of the rabbit
                    x[i, :] = (rabbitLocation - x[i, :]) - escapingEnergy * abs(
                        jumpStrength * rabbitLocation - x[i, :]
                    )

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                    r < 0.5 and abs(escapingEnergy) >= 0.5
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    jumpStrength = 2 * (1 - random.random())
                    x1 = rabbitLocation - escapingEnergy * abs(
                        jumpStrength * rabbitLocation - x[i, :]
                    )
                    x1 = numpy.clip(x1, lb, ub)

                    if objf(x1) < fitness:  # improved move?
                        x[i, :] = x1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        x2 = (
                            rabbitLocation
                            - escapingEnergy
                            * abs(jumpStrength * rabbitLocation - x[i, :])
                            + numpy.multiply(numpy.random.randn(dim), levy(dim))
                        )
                        x2 = numpy.clip(x2, lb, ub)
                        if objf(x2) < fitness:
                            x[i, :] = x2.copy()
                if (
                    r < 0.5 and abs(escapingEnergy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    jumpStrength = 2 * (1 - random.random())
                    x1 = rabbitLocation - escapingEnergy * abs(
                        jumpStrength * rabbitLocation - x.mean(0)
                    )
                    x1 = numpy.clip(x1, lb, ub)

                    if objf(x1) < fitness:  # improved move?
                        x[i, :] = x1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        x2 = (
                            rabbitLocation
                            - escapingEnergy
                            * abs(jumpStrength * rabbitLocation - x.mean(0))
                            + numpy.multiply(numpy.random.randn(dim), levy(dim))
                        )
                        x2 = numpy.clip(x2, lb, ub)
                        if objf(x2) < fitness:
                            x[i, :] = x2.copy()

        convergenceCurve[t] = rabbitEnergy
        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(rabbitEnergy)
                ]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergenceCurve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = rabbitEnergy
    s.bestIndividual = rabbitLocation

    return s


def levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step
