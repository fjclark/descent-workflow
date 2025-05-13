import torch
import typer
import numpy as np


def main(tops_and_ff_file: str) -> None:
    """
    Get the scales for the force field parameters in the given topology and force field file.
    """
    # Load the topology and force field
    loaded = torch.load(tops_and_ff_file)
    if type(loaded) is tuple:
        ff, _ = loaded
    else:
        ff = loaded

    ff = ff.to("cpu")

    for potential in ff.potentials:
        print(f"Potential: {potential.type}")
        for i, col_name in enumerate(potential.parameter_cols):
            # Get the mean value and std over all the parameters
            params = np.array([p[i].detach().numpy() for p in potential.parameters])
            mean = np.mean(params)
            std = np.std(params)
            scale = 1 / mean
            print(
                f"  {col_name}: mean = {mean:.4f}, std = {std:.4f}, scale = {scale:.4f}"
            )


if __name__ == "__main__":
    typer.run(main)
