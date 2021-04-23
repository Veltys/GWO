# -*- coding: utf-8 -*-


"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""


import csv
from pathlib import Path
import time
import warnings

import numpy

import benchmarks
import optimizers.BAT as bat
import optimizers.CS as cs
import optimizers.DE as de
import optimizers.FFA as ffa
import optimizers.GA as ga
import optimizers.GWO as gwo
import optimizers.HHO as hho
import optimizers.JAYA as jaya
import optimizers.MFO as mfo
import optimizers.MVO as mvo
import optimizers.PSO as pso
import optimizers.SCA as sca
import optimizers.SSA as ssa
import optimizers.WOA as woa
import plot_boxplot as box_plot
import plot_convergence as conv_plot


warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == "SSA":
        x = ssa.SSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "PSO":
        x = pso.PSO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GA":
        x = ga.GA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "BAT":
        x = bat.BAT(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "FFA":
        x = ffa.FFA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GWO":
        x = gwo.GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "WOA":
        x = woa.WOA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MVO":
        x = mvo.MVO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MFO":
        x = mfo.MFO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CS":
        x = cs.CS(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO":
        x = hho.HHO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SCA":
        x = sca.SCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "JAYA":
        x = jaya.JAYA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "DE":
        x = de.DE(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    else:
        return None
    return x


def run(optimizers, objectiveFuncs, NumOfRuns, params, exportFlags):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizers : list
        The list of optimizers names
    objectiveFuncs : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (populationSize)
        2. The number of iterations (iterations)
    exportFlags : set
        The set of Boolean flags which are:
        1. export (Exporting the results in a file)
        2. exportDetails (Exporting the detailed results in files)
        3. exportConvergence (Exporting the covergence plots)
        4. exportBoxplot (Exporting the box plots)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    populationSize = params["populationSize"]
    iterations = params["iterations"]

    # export results ?
    export = exportFlags["exportAvg"]
    exportDetails = exportFlags["exportDetails"]
    exportConvergence = exportFlags["exportConvergence"]
    exportBoxplot = exportFlags["exportBoxplot"]

    flag = False
    flagDetails = False

    # CSV Header for for the convergence
    cnvgHeader = []

    resultsDirectory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(resultsDirectory).mkdir(parents = True, exist_ok = True)

    for l in range(0, iterations):
        cnvgHeader.append("Iter" + str(l + 1))

    for _, optimizer in enumerate(optimizers):
        for _, objectiveFunc in enumerate(objectiveFuncs):
            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            for k in range(0, NumOfRuns):
                func_details = benchmarks.getFunctionDetails(objectiveFunc)
                x = selector(optimizer, func_details, populationSize, iterations)
                convergence[k] = x.convergence
                optimizerName = x.optimizer
                objfname = x.objfname
                if exportDetails == True:
                    exportToFile = resultsDirectory + "experiment_details.csv"
                    with open(exportToFile, "a", newline = "\n") as out:
                        writer = csv.writer(out, delimiter = ",")
                        if (
                            flagDetails == False
                        ):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "objfname", "ExecutionTime"], cnvgHeader]
                            )
                            writer.writerow(header)
                            flagDetails = True  # at least one experiment
                        executionTime[k] = x.executionTime
                        a = numpy.concatenate(
                            [[x.optimizer, x.objfname, x.executionTime], x.convergence]
                        )
                        writer.writerow(a)
                    out.close()

            if export == True:
                exportToFile = resultsDirectory + "experiment.csv"

                with open(exportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        flag == False
                    ):  # just one time to write the header of the CSV file
                        header = numpy.concatenate(
                            [["Optimizer", "objfname", "ExecutionTime"], cnvgHeader]
                        )
                        writer.writerow(header)
                        flag = True

                    avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
                    avgConvergence = numpy.around(
                        numpy.mean(convergence, axis = 0, dtype = numpy.float64), decimals = 2
                    ).tolist()
                    a = numpy.concatenate(
                        [[optimizerName, objfname, avgExecutionTime], avgConvergence]
                    )
                    writer.writerow(a)
                out.close()

    if exportConvergence == True:
        conv_plot.run(resultsDirectory, optimizers, objectiveFuncs, iterations)

    if exportBoxplot == True:
        box_plot.run(resultsDirectory, optimizers, objectiveFuncs, iterations)

    if flag == False:  # Faild to run at least one experiment
        print(
            "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
