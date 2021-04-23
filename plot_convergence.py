import matplotlib.pyplot as plt
import pandas as pd


def run(results_directory, optimizers, objectiveFuncs, iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment.csv")

    for _, objectivefunc in enumerate(objectiveFuncs):
        objective_name = objectivefunc

        startIteration = 0
        if "SSA" in optimizers:
            startIteration = 1
        allGenerations = [x + 1 for x in range(startIteration, iterations)]
        for _, optimizer in enumerate(optimizers):
            optimizer_name = optimizer

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            row = row.iloc[:, 3 + startIteration :]
            plt.plot(allGenerations, row.values.tolist()[0], label = optimizer_name)
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
        plt.grid()
        fig_name = results_directory + "/convergence-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
        # plt.show()
