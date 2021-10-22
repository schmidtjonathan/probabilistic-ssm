import numpy as np

C_U = "lightslategrey"
C_V = "darkgrey"
C_W = "C0"

C_I = "C16"
C_R = np.array([22, 168, 32]) / 255.0
C_D = np.array([245.0, 84.0, 66.0]) / 255.0

C_vdp_mu = "orangered"
C_sird_beta = "C7"
C_lv_alpha = "dodgerblue"

C_MAP = "tab:gray"

sample_style_args = dict(
    color="C0",
    lw=0.5,
    alpha=0.8,
    linestyle="dotted",
    zorder=-1,
)

MAP_style_args = dict(linestyle=(0, (2, 2)), color=C_MAP, zorder=2, lw=1.0)
gt_style_args = dict(color="black", linestyle="dotted", zorder=1, lw=0.5)


data_train_style_args = [
    dict(
        linestyle="",
        marker="o",
        markeredgecolor="white",
        markersize=3,
        markeredgewidth=0.5,
        markerfacecolor=C_U,
        zorder=13,
        clip_on=False,
    ),
    dict(
        linestyle="",
        marker="X",
        markeredgecolor="white",
        markersize=3,
        markeredgewidth=0.5,
        markerfacecolor=C_V,
        zorder=12,
        clip_on=False,
    ),
    dict(
        linestyle="",
        marker="D",
        markeredgecolor="white",
        markersize=3,
        markeredgewidth=0.5,
        markerfacecolor=C_W,
        zorder=11,
        clip_on=False,
    ),
]

data_val_style_args = [
    dict(
        linestyle="",
        marker="o",
        markeredgecolor=C_U,
        markersize=3,
        markeredgewidth=0.5,
        markerfacecolor="white",
        zorder=10,
    ),
    dict(
        linestyle="",
        marker="X",
        markeredgecolor=C_V,
        markersize=3,
        markeredgewidth=0.5,
        markerfacecolor="white",
        zorder=10,
    ),
    dict(
        linestyle="",
        marker="D",
        markeredgecolor=C_W,
        markersize=3,
        markeredgewidth=0.5,
        markerfacecolor="white",
        zorder=10,
    ),
]
