import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import yaml

from os import mkdir
from os.path import join
from datetime import datetime


def make_report(
    path: str,
    run_params: dict,
    results_dict: dict,
    run_time: int,
    best_solution: np.ndarray,
    best_cost: int,
) -> None:
    date_format = "%Y-%m-%d_%H-%M-%S"
    date_str = datetime.fromtimestamp(run_time).strftime(date_format)

    dir_path = path + "_" + date_str

    mkdir(dir_path)

    df_results = pd.DataFrame(results_dict)
    yaml_string = yaml.dump(run_params)

    df_results.to_csv(join(dir_path, "results.csv"), index=False)

    with open(join(dir_path, "config.yaml"), "w") as f:
        f.write(yaml_string)

    with open(join(dir_path, "solution.txt"), "w") as f:
        f.write(np.array2string(best_solution))

    with open(join(dir_path, "best_cost.txt"), "w") as f:
        f.write(str(best_cost))

    make_plots(dir_path=dir_path, df_results=df_results)


def make_plots(dir_path: str, df_results: pd.DataFrame) -> None:
    # cost plot
    cost_plot = sns.lineplot(data=df_results, x="iteration", y="best_cost")
    cost_plot.set(title="Best cost in time")

    cost_fig = cost_plot.get_figure()
    cost_fig.savefig(join(dir_path, "cost_plot.png"))

    plt.clf()

    # iteration time
    iter_time_plot = sns.lineplot(data=df_results, x="iteration", y="iteration_time")
    iter_time_plot.set(title="Time of each iteration")

    iter_time_fig = iter_time_plot.get_figure()
    iter_time_fig.savefig(join(dir_path, "iter_time_plot.png"))

    plt.clf()

    # number of groups
    no_groups_plot = sns.lineplot(data=df_results, x="iteration", y="number_of_groups")
    no_groups_plot.set(title="Number of groups in time")

    no_groups_fig = no_groups_plot.get_figure()
    no_groups_fig.savefig(join(dir_path, "no_grups_in_time.png"))

    plt.clf()

    # phases time
    phase_plot = sns.lineplot(
        data=df_results[
            [
                "local_leader_phase_time",
                "global_leader_phase_time",
                "local_leader_learning",
                "global_leader_learning",
                "local_leader_decision_phase",
                "global_leader_decision_phase",
            ]
        ]
    )
    phase_plot.set(title="Time of algorithm phases (in seconds)")

    phase_fig = phase_plot.get_figure()
    phase_fig.savefig(join(dir_path, "phases_time.png"))
