import random
import numpy
import time
from solution import solution


# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
def DE(objf, lb, ub, dim, PopSize, iters):

    mutation_factor = 0.5
    crossover_ratio = 0.7
    stoppingFunc = None

    # convert lb, ub to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]

    # solution
    s = solution()

    s.best = float("inf")

    # initialize population
    population = []

    populationFitness = numpy.array([float("inf") for _ in range(PopSize)])

    for p in range(PopSize):
        sol = []
        for d in range(dim):
            dVal = random.uniform(lb[d], ub[d])
            sol.append(dVal)

        population.append(sol)

    population = numpy.array(population)

    # calculate fitness for all the population
    for i in range(PopSize):
        fitness = objf(population[i, :])
        populationFitness[p] = fitness
        # s.func_evals += 1

        # is leader ?
        if fitness < s.best:
            s.best = fitness
            s.leader_solution = population[i, :]

    convergenceCurve = numpy.zeros(iters)
    # start work
    print('DE is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0
    while t < iters:
        # should i stop
        if stoppingFunc is not None and stoppingFunc(s.best, s.leader_solution, t):
            break

        # loop through population
        for i in range(PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            idsExceptCurrent = [_ for _ in range(PopSize) if _ != i]
            id1, id2, id3 = random.sample(idsExceptCurrent, 3)

            mutantSol = []
            for d in range(dim):
                dVal = population[id1, d] + mutation_factor * (
                    population[id2, d] - population[id3, d]
                )

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    dVal = population[i, d]

                # add dimension value to the mutant solution
                mutantSol.append(dVal)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutantSol = numpy.clip(mutantSol, lb, ub)

            # calc fitness
            mutantFitness = objf(mutantSol)
            # s.func_evals += 1

            # replace if mutantFitness is better
            if mutantFitness < populationFitness[i]:
                population[i, :] = mutantSol
                populationFitness[i] = mutantFitness

                # update leader
                if mutantFitness < s.best:
                    s.best = mutantFitness
                    s.leader_solution = mutantSol

        convergenceCurve[t] = s.best
        if t % 1 == 0:
            print(
                ["At iteration " + str(t + 1) + " the best fitness is " + str(s.best)]
            )

        # increase iterations
        t = t + 1

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergenceCurve
        s.optimizer = "DE"
        s.objfname = objf.__name__

    # return solution
    return s
