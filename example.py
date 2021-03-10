#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""


import getopt
import sys

import Config
from optimizer import run


def main(argv):
    cnf = Config.Config()

    try:
        opts, _ = getopt.getopt(
            argv,
            'b:d:v',
            [
                'bench=',
                'dim=',
            ]
        )

    except getopt.GetoptError:
        print('Usage: example.py [-b benchmark_id or --bench=benchmark_id] [-d dimensions or --dim=dimensions] [-v]')

        sys.exit(2)


    for opt, arg in opts:
        if opt in ('-b', '--bench'):
            cnf.benchmark = int(arg)

        elif opt in ('-d', '--dim'):
            cnf.dimensions = int(arg)

        elif opt in ('-v'):
            cnf.verbosity = True


    # Select optimizers
    # "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
    optimizer = ["GWO"]

    # Select benchmark function"
    # "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"
    # "Ca1","Ca2","Gt1","Mes","Mef","Sag","Tan","Ros"
    objectiveFuncs = ["F24"]

    # Select number of repetitions for each experiment.
    # To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
    NumOfRuns = 30

    # Select general parameters for all optimizers (population size, number of iterations) ....
    params = {"populationSize": 30, "iterations": 1500}

    # Choose whether to Export the results in different formats
    exportFlags = {
        "exportAvg": True,
        "exportDetails": True,
        "exportConvergence": True,
        "exportBoxplot": True,
    }

    run(optimizer, objectiveFuncs, NumOfRuns, params, exportFlags)


if __name__ == '__main__':
    main(sys.argv[1:])
