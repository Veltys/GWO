import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(results_directory, optimizers, objectiveFuncs, iterations):
    plt.ioff()

    fileResultsDetailsData = pd.read_csv(results_directory + "/experiment_details.csv")
    for _, objectiveFunc in enumerate(objectiveFuncs):

        # Box Plot
        data = []

        for _, optimizer in enumerate(optimizers):
            objective_name = objectiveFunc
            optimizer_name = optimizer

            detailedData = fileResultsDetailsData[
                (fileResultsDetailsData["Optimizer"] == optimizer_name)
                & (fileResultsDetailsData["objfname"] == objective_name)
            ]
            detailedData = detailedData["Iter" + str(iterations)]
            detailedData = np.array(detailedData).T.tolist()
            data.append(detailedData)

        # , notch = True
        box = plt.boxplot(data, patch_artist = True, labels = optimizers)

        colors = [
            "#5c9eb7",
            "#f77199",
            "#cf81d2",
            "#4a5e6a",
            "#f45b18",
            "#ffbd35",
            "#6ba5a1",
            "#fcd1a1",
            "#c3ffc1",
            "#68549d",
            "#1c8c44",
            "#a44c40",
            "#404636",
        ]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        plt.legend(
            handles = box["boxes"],
            labels = optimizers,
            loc = "upper right",
            bbox_to_anchor = (1.2, 1.02),
        )
        fig_name = results_directory + "/boxplot-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches = "tight")
        plt.clf()
        # plt.show()
