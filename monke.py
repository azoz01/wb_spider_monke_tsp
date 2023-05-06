import typer
import yaml

from typing_extensions import Annotated

from engine.monke_optimizer import MonkeOptimizer
from engine.tsp import TspWrapper

app = typer.Typer()


@app.command(
    help="Optimize traveling salesman problem using spider-monkey discrete optimization."
)
def main(
    problem_path: Annotated[str, typer.Argument(..., help="Path to problem.")],
    config_path: Annotated[
        str, typer.Option(..., help="Path to configuration of optimizer.")
    ] = "config/optmizer_config.yaml",
    n_iter: Annotated[
        int, typer.Option(..., help="Number of iterations to perform.")
    ] = 10_000,
    timeout_seconds: Annotated[
        int, typer.Option(..., help="Maximum optimization time")
    ] = 300,
):
    problem = TspWrapper.from_atsp_full_matrix(problem_path)

    with open(config_path) as f:
        optimizer_config = yaml.load(f, Loader=yaml.CLoader)
    optimizer = MonkeOptimizer(**optimizer_config)
    solution, best_cost = optimizer.optimize(
        problem, timeout_seconds=timeout_seconds, n_iter=n_iter
    )
    print(f"{solution=}")
    print(f"{best_cost=}")


if __name__ == "__main__":
    app()
