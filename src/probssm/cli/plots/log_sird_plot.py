import argparse
import functools
import json
import logging
import pathlib
from datetime import datetime, timedelta

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import probssm
import probssm.plotting as ssmplot
import scipy.special
from matplotlib.legend_handler import HandlerTuple
from pandas import to_datetime

from ._style import *

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - plotting - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("logdir", type=pathlib.Path)

    parser.add_argument(
        "--smoothing-or-filter",
        type=str,
        choices=["smoothing", "filter"],
        default="smoothing",
    )

    parser.add_argument("--out-dir", type=pathlib.Path, required=False)
    parser.add_argument("--out-filename", type=str, default="figure")
    parser.add_argument(
        "--out-format", type=str, default="pdf", choices=["pdf", "png", "pgf"]
    )

    return parser.parse_args()


def main():

    args = parse_args()

    log_dir = args.logdir
    assert log_dir.is_dir()

    _style_file = pathlib.Path(__file__).parent.resolve() / "sirmodel.mplstyle"
    assert _style_file.is_file(), f"{_style_file} does not exist."
    plt.style.use(str(_style_file))

    if args.out_format == "pgf":
        matplotlib.use("pgf")

    if args.out_format == "png":
        plt.rcParams["text.usetex"] = False

    sample_style_args = dict(
        color="C0",
        lw=0.8,
        alpha=0.8,
        linestyle="--",
        zorder=-1,
    )

    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    figure = plt.figure(figsize=(8, 3), constrained_layout=True)
    gs = figure.add_gridspec(2, 1)

    # ##################################################################################
    # BOTH
    # ##################################################################################

    ax11 = figure.add_subplot(gs[0, 0])
    ax21 = figure.add_subplot(gs[1, 0], sharex=ax11)

    # Load info from disk
    projections = np.load(log_dir / "projections.npz")
    data_dict = np.load(log_dir / "data_info.npz")
    with open(log_dir / "info.json", "r") as file_handle:
        info = json.load(file_handle)

    proj_to_beta = projections["E0_beta"] @ projections["E_beta"]
    proj_to_ird = (projections["E0_x"] @ projections["E_x"])[1:, :]

    sigmoid_x_offset = -scipy.special.logit(info["beta_prior_mean"])
    sigmoid_transform = functools.partial(
        probssm.sloped_sigmoid,
        slope=info["sigmoid_slope"],
        x_offset=sigmoid_x_offset,
    )

    SIRD_data = data_dict["sird_data"]

    dense_grid = data_dict["dense_grid"]
    day_zero = to_datetime(data_dict["day_zero"])
    date_range_x = np.array([to_datetime(ts) for ts in data_dict["date_range_x"]])

    plot_every = int(1.0 / info["filter_step_size"])
    idcs = np.arange(0, len(dense_grid), dtype=int)[::plot_every]

    # Convert x axis to dates
    ax11 = ssmplot.x_axis_to_date(ax11)
    ax21, ax21_date_locations = ssmplot.x_axis_to_date(
        ax21, day_zero=day_zero, locations=dense_grid
    )

    # INFECTIOUS

    ax11, ax11datai = ssmplot.inject_data(
        ax11,
        data=SIRD_data[:, 1:2],
        locations=date_range_x,
        train_idcs=data_dict["train_idcs"],
        val_idcs=data_dict["val_idcs"],
        plot_every=10,
        labels=["I data"],
        train_ax_kwargs=data_train_style_args[0],
        val_ax_kwargs=data_val_style_args[0],
    )

    ax11 = ssmplot.inject_posterior_samples(
        ax11,
        log_dir=log_dir,
        projmat=proj_to_ird[0:1, :],
        locations=ax21_date_locations,
        idcs=idcs,
        transform_fn=np.exp,
        sample_ax_kwargs=sample_style_args,
    )

    ax11, ax11meani = ssmplot.inject_posterior_mean(
        ax11,
        log_dir=log_dir,
        projmat=proj_to_ird[0:1, :],
        locations=ax21_date_locations,
        idcs=idcs,
        smoothing_or_filter=args.smoothing_or_filter,
        logfile_suffix="_first",
        labels=["posterior I"],
        transform_fn=np.exp,
        mean_ax_kwargs=dict(color=C_U),
    )

    ax11, _ = ssmplot.inject_posterior_credible_interval(
        ax11,
        log_dir=log_dir,
        projmat=proj_to_ird[0:1, :],
        locations=ax21_date_locations,
        idcs=idcs,
        smoothing_or_filter=args.smoothing_or_filter,
        logfile_suffix="_first",
        transform_fn=lambda s: np.minimum(np.exp(s), 1e6),
        ci_ax_kwargs=dict(color=C_U, alpha=0.2),
    )

    # BETA

    ax21 = ssmplot.inject_posterior_samples(
        ax21,
        log_dir=log_dir,
        projmat=proj_to_beta,
        locations=ax21_date_locations,
        idcs=idcs,
        transform_fn=sigmoid_transform,
        sample_ax_kwargs=sample_style_args,
    )

    ax21, betahandle = ssmplot.inject_posterior_mean(
        ax21,
        log_dir=log_dir,
        projmat=proj_to_beta,
        locations=ax21_date_locations,
        idcs=idcs,
        smoothing_or_filter=args.smoothing_or_filter,
        logfile_suffix="_first",
        labels=["posterior mean"],
        transform_fn=sigmoid_transform,
        mean_ax_kwargs=dict(color=C_sird_beta),
    )
    # ax21, mapbetahandle = ssmplot.inject_posterior_mean(
    #     ax21,
    #     log_dir=log_dir,
    #     projmat=proj_to_beta,
    #     locations=ax21_date_locations,
    #     idcs=idcs,
    #     smoothing_or_filter=args.smoothing_or_filter,
    #     logfile_suffix="_last",
    #     transform_fn=sigmoid_transform,
    #     mean_ax_kwargs=MAP_style_args,
    # )
    ax21, _ = ssmplot.inject_posterior_credible_interval(
        ax21,
        log_dir=log_dir,
        projmat=proj_to_beta,
        locations=ax21_date_locations,
        idcs=idcs,
        smoothing_or_filter=args.smoothing_or_filter,
        logfile_suffix="_first",
        transform_fn=sigmoid_transform,
        ci_ax_kwargs=dict(color=C_sird_beta, alpha=0.2),
    )

    march22nd = datetime.strptime("2020-03-22 00:00:00", "%Y-%m-%d %H:%M:%S")
    march22nd_index = list(date_range_x).index(march22nd)

    may6th = datetime.strptime("2020-05-06 00:00:00", "%Y-%m-%d %H:%M:%S")
    may6th_index = list(date_range_x).index(may6th)

    october7th = datetime.strptime("2020-10-07 00:00:00", "%Y-%m-%d %H:%M:%S")
    october7th_index = list(date_range_x).index(october7th)

    november2nd = datetime.strptime("2020-11-02 00:00:00", "%Y-%m-%d %H:%M:%S")
    november2nd_index = list(date_range_x).index(november2nd)

    december16th = datetime.strptime("2020-12-16 00:00:00", "%Y-%m-%d %H:%M:%S")
    december16th_index = list(date_range_x).index(december16th)

    december24th = datetime.strptime("2020-12-24 00:00:00", "%Y-%m-%d %H:%M:%S")
    december24th_index = list(date_range_x).index(december24th)

    ax21.annotate(
        "",
        xy=(october7th - timedelta(days=5), 1.0),
        xytext=(may6th, 1.0),
        arrowprops=dict(
            arrowstyle="-|>", color="C0", linewidth=plt.rcParams["xtick.major.width"]
        ),
    )

    ax21 = ssmplot.inject_dates(
        ax21,
        [march22nd, may6th, october7th, november2nd, december16th],
        labels=["1", "2", "3", "4", "5"],
        lw=plt.rcParams["xtick.major.width"],
        color="C0",
        alpha=1.0,
        linestyle="--",
        ymax=0.47,
        ymin=-0.1,
        zorder=-15,
    )

    # Finalize
    xticks = np.arange(
        start=mdates.date2num(day_zero),
        stop=mdates.date2num(day_zero) + data_dict["time_domain"][1],
        step=90,
    )

    ax11.set_ylim([-0.5, 6])
    ax11.set_yticks([0, 5])
    ax21.set_ylim([-0.015, 0.59])

    ax11.margins(0.0)
    ax21.margins(0.0)

    ax11.set_ylabel("Counts [cpt]")
    ax21.set_ylabel(r"$\beta$")
    ax21.set_xticks(xticks)

    plt.setp(ax11.get_xticklabels(), visible=False)

    figure.legend(
        [tuple(ax11datai[0]), ax11meani[0], betahandle[0]],  # , mapbetahandle[0]],
        [
            "Data (train/val)",
            "Posterior infectious",
            "Posterior contact rate",
            "MAP estimate",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="center left",
        bbox_to_anchor=(1.0, 0.55),
        frameon=False,
    )

    figure.align_ylabels()

    if args.out_dir is not None:
        _figure_save_path = args.out_dir / f"{args.out_filename}.{args.out_format}"
        figure.savefig(_figure_save_path, bbox_inches="tight")

        logging.info(f"Saved figure to {_figure_save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
