import argparse
import functools
import json
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from matplotlib.legend_handler import HandlerTuple

import probssm.plotting as ssmplot

from ._style import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("logdir", type=pathlib.Path)

    parser.add_argument("--out-dir", type=pathlib.Path, required=False)
    parser.add_argument("--out-filename", type=str, default="figure")
    parser.add_argument(
        "--out-format", type=str, default="png", choices=["pdf", "png", "pgf"]
    )

    return parser.parse_args()


def main():

    args = parse_args()

    log_dir = args.logdir
    assert log_dir.is_dir()

    _style_file = pathlib.Path(__file__).parent.resolve() / "sirmodel.mplstyle"
    assert _style_file.is_file(), f"{_style_file} does not exist."
    plt.style.use(str(_style_file))

    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["axes.titlesize"] = 7
    plt.rcParams["xtick.labelsize"] = 5.5
    plt.rcParams["ytick.labelsize"] = 5.5
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    figure = plt.figure(figsize=(5.5, 3.0), constrained_layout=True)
    gs = figure.add_gridspec(2, 4)

    axlvstate = figure.add_subplot(gs[:, 0:2])
    axlvparam1 = figure.add_subplot(gs[0, 2])
    axlvparam2 = figure.add_subplot(gs[1, 2], sharex=axlvparam1)
    axlvparam3 = figure.add_subplot(gs[0, 3])
    axlvparam4 = figure.add_subplot(gs[1, 3], sharex=axlvparam3)

    #                                                                                 LV
    # ##################################################################################

    # Load info from disk
    projections = np.load(log_dir / "projections.npz")
    data_dict = np.load(log_dir / "data_info.npz")
    with open(log_dir / "info.json", "r") as file_handle:
        info = json.load(file_handle)

    proj_to_uv = projections["E0_x"] @ projections["E_x"]
    proj_to_alpha = projections["E0_alpha"] @ projections["E_alpha"]
    proj_to_beta = projections["E0_beta"] @ projections["E_beta"]
    proj_to_gamma = projections["E0_gamma"] @ projections["E_gamma"]
    proj_to_delta = projections["E0_delta"] @ projections["E_delta"]

    data = data_dict["data"]
    gt_data = data_dict["gt_data"]
    gt_alpha = data_dict["gt_alpha"]
    gt_beta = data_dict["gt_beta"]
    gt_gamma = data_dict["gt_gamma"]
    gt_delta = data_dict["gt_delta"]

    data_grid = data_dict["data_grid"]
    dense_grid = data_dict["dense_grid"]

    plot_every = 1  # int(1.0 / info["filter_step_size"])
    idcs = np.arange(0, len(dense_grid), dtype=int)[::plot_every]

    alpha_offset = -np.log(info["alpha_prior_mean"])
    beta_offset = -np.log(info["beta_prior_mean"])
    gamma_offset = -np.log(info["gamma_prior_mean"])
    delta_offset = -np.log(info["delta_prior_mean"])
    alpha_link_fn = lambda x: np.exp(x - alpha_offset)
    beta_link_fn = lambda x: np.exp(x - beta_offset)
    gamma_link_fn = lambda x: np.exp(x - gamma_offset)
    delta_link_fn = lambda x: np.exp(x - delta_offset)

    axlvstate, lvdatahandles = ssmplot.inject_data(
        axlvstate,
        data=data,
        locations=data_grid,
        train_idcs=data_dict["train_idcs"],
        val_idcs=data_dict["val_idcs"],
        plot_every=3,
        labels=["U data", "V data"],
        train_ax_kwargs=data_train_style_args[:2],
    )

    axlvstate, _ = ssmplot.inject_groundtruth(
        axlvstate,
        gt_data=gt_data,
        locations=dense_grid,
        idcs=idcs,
        gt_ax_kwargs=gt_style_args,
    )

    axlvstate, _ = ssmplot.inject_posterior_mean(
        axlvstate,
        log_dir=log_dir,
        projmat=proj_to_uv,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=None,
        mean_ax_kwargs=[dict(color=C_U), dict(color=C_V)],
    )

    axlvstate, _ = ssmplot.inject_posterior_credible_interval(
        axlvstate,
        log_dir=log_dir,
        projmat=proj_to_uv,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=None,
        ci_ax_kwargs=[
            dict(color=C_U, alpha=0.2, linewidth=0.0),
            dict(color=C_V, alpha=0.2, linewidth=0.0),
        ],
    )

    # ALPHA

    axlvparam1, lvparam1meanhandles = ssmplot.inject_posterior_mean(
        axlvparam1,
        log_dir=log_dir,
        projmat=proj_to_alpha,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=alpha_link_fn,
        mean_ax_kwargs=dict(color=C_lv_alpha),
    )
    axlvparam1, _ = ssmplot.inject_posterior_credible_interval(
        axlvparam1,
        log_dir=log_dir,
        projmat=proj_to_alpha,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=alpha_link_fn,
        ci_ax_kwargs=dict(color=C_lv_alpha, alpha=0.2, linewidth=0.0),
    )

    axlvparam1, _ = ssmplot.inject_posterior_mean(
        axlvparam1,
        log_dir=log_dir,
        projmat=proj_to_alpha,
        locations=dense_grid,
        idcs=idcs,
        logfile_suffix="_map",
        transform_fn=alpha_link_fn,
        mean_ax_kwargs=MAP_style_args,
    )

    axlvparam1, _ = ssmplot.inject_groundtruth(
        axlvparam1,
        gt_data=gt_alpha,
        locations=dense_grid,
        idcs=idcs,
        gt_ax_kwargs=gt_style_args,
    )

    # BETA

    axlvparam2, _ = ssmplot.inject_posterior_mean(
        axlvparam2,
        log_dir=log_dir,
        projmat=proj_to_beta,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=beta_link_fn,
        mean_ax_kwargs=dict(color=C_lv_alpha),
    )
    axlvparam2, _ = ssmplot.inject_posterior_credible_interval(
        axlvparam2,
        log_dir=log_dir,
        projmat=proj_to_beta,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=beta_link_fn,
        ci_ax_kwargs=dict(color=C_lv_alpha, alpha=0.2, linewidth=0.0),
    )

    axlvparam2, _ = ssmplot.inject_posterior_mean(
        axlvparam2,
        log_dir=log_dir,
        projmat=proj_to_beta,
        locations=dense_grid,
        idcs=idcs,
        logfile_suffix="_map",
        transform_fn=beta_link_fn,
        mean_ax_kwargs=MAP_style_args,
    )

    axlvparam2, _ = ssmplot.inject_groundtruth(
        axlvparam2,
        gt_data=gt_beta,
        locations=dense_grid,
        idcs=idcs,
        gt_ax_kwargs=gt_style_args,
    )

    # GAMMA

    axlvparam3, _ = ssmplot.inject_posterior_mean(
        axlvparam3,
        log_dir=log_dir,
        projmat=proj_to_gamma,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=gamma_link_fn,
        mean_ax_kwargs=dict(color=C_lv_alpha),
    )
    axlvparam3, _ = ssmplot.inject_posterior_credible_interval(
        axlvparam3,
        log_dir=log_dir,
        projmat=proj_to_gamma,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=gamma_link_fn,
        ci_ax_kwargs=dict(color=C_lv_alpha, alpha=0.2, linewidth=0.0),
    )

    axlvparam3, _ = ssmplot.inject_posterior_mean(
        axlvparam3,
        log_dir=log_dir,
        projmat=proj_to_gamma,
        locations=dense_grid,
        idcs=idcs,
        logfile_suffix="_map",
        transform_fn=gamma_link_fn,
        mean_ax_kwargs=MAP_style_args,
    )

    axlvparam3, _ = ssmplot.inject_groundtruth(
        axlvparam3,
        gt_data=gt_gamma,
        locations=dense_grid,
        idcs=idcs,
        gt_ax_kwargs=gt_style_args,
    )

    # DELTA

    axlvparam4, _ = ssmplot.inject_posterior_mean(
        axlvparam4,
        log_dir=log_dir,
        projmat=proj_to_delta,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=delta_link_fn,
        mean_ax_kwargs=dict(color=C_lv_alpha),
    )
    axlvparam4, _ = ssmplot.inject_posterior_credible_interval(
        axlvparam4,
        log_dir=log_dir,
        projmat=proj_to_delta,
        locations=dense_grid,
        idcs=idcs,
        transform_fn=delta_link_fn,
        ci_ax_kwargs=dict(color=C_lv_alpha, alpha=0.2, linewidth=0.0),
    )

    axlvparam4, lvdeltamaphandle = ssmplot.inject_posterior_mean(
        axlvparam4,
        log_dir=log_dir,
        projmat=proj_to_delta,
        locations=dense_grid,
        idcs=idcs,
        logfile_suffix="_map",
        transform_fn=delta_link_fn,
        mean_ax_kwargs=MAP_style_args,
    )

    axlvparam4, lvgthandles = ssmplot.inject_groundtruth(
        axlvparam4,
        gt_data=gt_delta,
        locations=dense_grid,
        idcs=idcs,
        gt_ax_kwargs=gt_style_args,
    )

    figure.legend(
        [
            tuple(lvdatahandles),
            (lvparam1meanhandles[0],),
            lvdeltamaphandle[0],
            lvgthandles[0],
        ],
        ["Data", "Posterior states", "Posterior parameters", "MAP estimate", "Truth"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="center left",
        bbox_to_anchor=(0.07, -0.05),
        ncol=5,
        frameon=False,
    )

    for ax in figure.axes:
        ax.spines["left"].set_position(("outward", 2))
        ax.spines["bottom"].set_position(("outward", 2))

    axlvstate.set_title("State trajectories")
    axlvparam1.set_title("Inferred parameters")

    # axlvstate.set_title("VDP", loc="left", fontweight="bold", ha="right")
    axlvstate.set_ylabel("VDP State")
    axlvstate.set_xticks([0, 12, 25])
    axlvstate.set_ylim([-6, 6])
    axlvstate.set_yticks([-6, 0, 6])
    axlvstate.set_xlabel("Time $t$")

    axlvstate.margins(0.0)
    axlvstate.set_ylim([0, 40])
    axlvstate.set_yticks([0, 20, 40])
    axlvstate.set_ylabel("LV Population")
    axlvparam1.set_ylabel(r"$a$")
    axlvparam2.set_ylabel(r"$b$")
    axlvparam3.set_ylabel(
        r"$c$",
    )
    axlvparam4.set_ylabel(
        r"$d$",
    )

    axlvstate.set_xlabel("Time $t$")
    axlvstate.set_xticks([0, 30, 60])
    axlvparam2.set_xlabel("Time $t$")
    axlvparam4.set_xlabel("Time $t$")
    axlvparam2.set_xticks([0, 30, 60])
    axlvparam4.set_xticks([0, 30, 60])

    axlvparam1.set_ylim([0.2, 0.6])
    axlvparam1.set_yticks([0.2, 0.6])
    axlvparam2.set_ylim([0.03, 0.06])
    axlvparam2.set_yticks([0.03, 0.06])
    axlvparam3.set_ylim([0.3, 1.0])
    axlvparam3.set_yticks([0.3, 1.0])
    axlvparam4.set_ylim([0.03, 0.06])
    axlvparam4.set_yticks([0.03, 0.06])

    plt.setp(axlvparam1.get_xticklabels(), visible=False)
    plt.setp(axlvparam3.get_xticklabels(), visible=False)
    # plt.setp(ax4.get_yticklabels(), visible=False)
    # plt.setp(ax5.get_yticklabels(), visible=False)

    figure.align_labels()

    if args.out_dir is not None:
        _figure_save_path = args.out_dir / f"{args.out_filename}.{args.out_format}"
        figure.savefig(_figure_save_path, bbox_inches="tight")

        logging.info(f"Saved figure to {_figure_save_path}")
    else:
        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
