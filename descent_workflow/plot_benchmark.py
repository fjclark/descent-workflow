"""From https://raw.githubusercontent.com/openforcefield/yammbs-dataset-submission/refs/heads/main/plot.py"""

import logging
import warnings
from pathlib import Path
from typing import Any

import click
import loguru
import numpy
import pandas
import seaborn as sea
from matplotlib import pyplot
from pandas import DataFrame as DF

pyplot.style.use("ggplot")

logger = loguru.logger

# try to suppress stereo warnings - from lily's valence-fitting
# curate-dataset.py
logging.getLogger("openff").setLevel(logging.ERROR)

# suppress divide by zero in numpy.log
warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)

pandas.set_option("display.max_columns", None)


def load_ids_to_remove(paths: list[Path] | None = None) -> list[int]:
    """Load the list of problematic record ids which should be removed."""
    if paths is None:
        paths = []
    ids_to_remove = []
    for path in paths:
        ids_to_remove.extend(numpy.loadtxt(path, dtype=int).tolist())
    return ids_to_remove


def load_bench(d: Path, ids_to_remove: list[int] | None = None) -> pandas.DataFrame:
    """Load the DDE, RMSD, TFD, and ICRMSD results from the CSV files in ``d``
    and return the result as a merged dataframe
    """
    dde = pandas.read_csv(d / "dde.csv")
    dde.columns = ["rec_id", "dde"]
    rmsd = pandas.read_csv(d / "rmsd.csv")
    rmsd.columns = ["rec_id", "rmsd"]
    tfd = pandas.read_csv(d / "tfd.csv")
    tfd.columns = ["rec_id", "tfd"]
    icrmsd = pandas.read_csv(d / "icrmsd.csv")
    icrmsd.columns = ["rec_id", "bonds", "angles", "dihedrals", "impropers"]
    ret = dde.merge(rmsd).pipe(DF.merge, tfd).pipe(DF.merge, icrmsd)
    # ret = dde.merge(rmsd).pipe(DF.merge, tfd)

    # remove any rows with rec_ids in ids_to_remove
    initial_shape = ret.shape
    if ids_to_remove is not None:
        ret = ret[~ret["rec_id"].isin(ids_to_remove)]
        assert isinstance(ret, pandas.DataFrame)
    final_shape = ret.shape
    logger.info(f"removed {initial_shape[0] - final_shape[0]} rows with problematic rec_ids")

    logger.info(f"loaded {ret.shape} rows for {d}")
    return ret


def load_benches(ffs: Any, ids_to_remove: list[int]) -> list[pandas.DataFrame]:
    return [load_bench(Path(ff), ids_to_remove) for ff in ffs]


def merge_metrics(dfs: list[pandas.DataFrame], names: list[str], metric: str) -> pandas.DataFrame:
    assert len(dfs) > 0, "must provide at least one dataframe"
    df = dfs[0][["rec_id", metric]].copy()
    df.columns = ["rec_id", names[0]]
    for i, d in enumerate(dfs[1:]):
        name = names[i + 1]
        to_add = d[["rec_id", metric]].copy()
        to_add.columns = ["rec_id", name]
        df = df.merge(to_add, on="rec_id")
    return df


def plot_ddes(dfs: list[pandas.DataFrame], names: list[str], out_dir: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(6, 4))
    ddes = merge_metrics(dfs, names, "dde")
    ax = sea.histplot(
        data=ddes.iloc[:, 1:],
        binrange=(-15, 15),
        bins=16,
        element="step",
        fill=False,
        alpha=0.8,
    )
    label = "DDE (kcal mol$^{-1}$)"
    # Get the auto-created legend from seaborn and move it outside the plot
    leg = axis.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1, 1.05))
    ax.set_xlabel(label)
    pyplot.savefig(f"{out_dir}/dde.png", dpi=300, bbox_inches="tight")
    pyplot.close()

    # Get absolute values of DDE
    abs_ddes = ddes.iloc[:, 1:].abs()
    # Plot cumulative distribution function (CDF)
    figure, axis = pyplot.subplots(figsize=(6, 4))
    sea.ecdfplot(data=abs_ddes, ax=axis)
    leg = axis.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.1, 0.9))
    axis.set_xlim((-0.1, 8))
    axis.set_xlabel("Absolute DDE (kcal mol$^{-1}$)")
    pyplot.savefig(f"{out_dir}/dde_cdf.png", dpi=300, bbox_inches="tight")
    pyplot.close()


def plot_rmsds(dfs: list[pandas.DataFrame], names: list[str], out_dir: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(6, 4))
    rmsds = merge_metrics(dfs, names, "rmsd")
    ax = sea.kdeplot(data=numpy.log10(rmsds.iloc[:, 1:]))
    ax.set_xlim((-2.0, 0.7))
    ax.set_xlabel("Log RMSD")
    # Get the auto-created legend from seaborn and move it outside the plot
    leg = axis.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1, 1.05))
    pyplot.savefig(f"{out_dir}/rmsd.png", dpi=300, bbox_inches="tight")
    pyplot.close()

    figure, axis = pyplot.subplots(figsize=(6, 4))
    for col in rmsds.columns[1:]:
        sea.ecdfplot(rmsds[col], label=col, ax=axis)
    axis.set_xlim((0, 1.25))
    axis.set_xlabel("RMSD (Å)")
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1.05))
    pyplot.savefig(f"{out_dir}/rmsd_cdf.png", dpi=300, bbox_inches="tight")
    pyplot.close()


def plot_tfds(dfs: list[pandas.DataFrame], names: list[str], out_dir: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(6, 4))
    tfds = merge_metrics(dfs, names, "tfd")
    ax = sea.kdeplot(data=numpy.log10(tfds.iloc[:, 1:]))
    ax.set_xlim((-4.0, 0.5))
    ax.set_xlabel("Log TFD")
    # Get the auto-created legend from seaborn and move it outside the plot
    leg = axis.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1, 1.05))
    pyplot.savefig(f"{out_dir}/tfd.png", dpi=300, bbox_inches="tight")
    pyplot.close()

    figure, axis = pyplot.subplots(figsize=(6, 4))
    for col in tfds.columns[1:]:
        sea.ecdfplot(tfds[col], label=col, ax=axis)
    ax = sea.ecdfplot(tfds.iloc[:, 1:], ax=axis)
    ax.set_xlim((0, 0.2))
    ax.set_xlabel("TFD")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1.05))
    pyplot.savefig(f"{out_dir}/tfd_cdf.png", dpi=300, bbox_inches="tight")
    pyplot.close()


def plot_icrmsds(dfs: list[pandas.DataFrame], names: list[str], out_dir: Path) -> None:
    titles = {
        "bonds": "Bond Internal Coordinate RMSDs",
        "angles": "Angle Internal Coordinate RMSDs",
        "dihedrals": "Proper Torsion Internal Coordinate RMSDs",
        "impropers": "Improper Torsion Internal Coordinate RMSDs",
    }
    ylabels = {
        "bonds": "Bond error (Å)",
        "angles": "Angle error (̂°)",
        "dihedrals": "Proper Torsion error (°)",
        "impropers": "Improper Torsion error(°)",
    }
    for m in ["bonds", "angles", "dihedrals", "impropers"]:
        full = merge_metrics(dfs, names, m)
        df = full.iloc[:, 1:]
        figure, axis = pyplot.subplots(figsize=(6, 4))
        ax = sea.boxplot(df)
        pyplot.title(titles[m])
        ax.set_ylabel(ylabels[m])
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        pyplot.savefig(f"{out_dir}/{m}.png", dpi=300, bbox_inches="tight")
        pyplot.close()


def plot(ffs: Any, out_dir: str, ids_to_remove_paths: list[Path] | None = None) -> None:
    """Plot each of the `dde`, `rmsd`, and `tfd` CSV files found in `ff/output`.

    For `ff` in `ffs` and write the resulting PNG images to out_dir.
    """
    if ids_to_remove_paths is None:
        ids_to_remove_paths = []
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True)

    ids_to_remove = load_ids_to_remove(ids_to_remove_paths)
    dfs = load_benches(ffs, ids_to_remove=ids_to_remove)

    names = [Path(ff).name for ff in ffs]

    for name, df in zip(names, dfs, strict=True):
        df.to_csv(f"{out_dir}/{name}.csv")

    plot_rmsds(dfs, names, out_dir_path)
    plot_ddes(dfs, names, out_dir_path)
    plot_tfds(dfs, names, out_dir_path)
    plot_icrmsds(dfs, names, out_dir_path)


@click.command()
@click.argument("forcefields", nargs=-1)
@click.option("--output_dir", "-o", default="/tmp")
@click.option(
    "--ids_to_remove_paths",
    "-i",
    multiple=True,
    type=click.Path(exists=True),
    help="Path to file containing problematic record ids to remove",
)
def main(
    forcefields: tuple[str, ...],
    output_dir: str,
    ids_to_remove_paths: tuple[str, ...] | None = None,
) -> None:
    if ids_to_remove_paths is None:
        ids_to_remove_paths = ()
    ids_to_remove_paths_list = [Path(p) for p in ids_to_remove_paths]
    plot(forcefields, output_dir, ids_to_remove_paths_list)


if __name__ == "__main__":
    main()
