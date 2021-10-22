import functools
import logging
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from scipy.integrate import solve_ivp


def x_axis_to_date(
    ax,
    day_zero=None,
    locations=None,
):

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    if locations is not None and day_zero is not None:
        date_locations = np.array(
            [day_zero + timedelta(days=float(d)) for d in locations]
        )
        return ax, date_locations

    return ax


def inject_data(
    ax,
    data,
    locations,
    train_idcs,
    val_idcs,
    plot_every=1,
    labels=None,
    train_ax_kwargs=None,
    val_ax_kwargs=None,
):
    valid_indices = np.arange(len(locations), dtype=np.int64)[::plot_every]
    train_idcs = train_idcs[np.where(np.isin(train_idcs, valid_indices))]
    val_idcs = val_idcs[np.where(np.isin(val_idcs, valid_indices))]
    train_locations = locations[train_idcs]
    val_locations = locations[val_idcs]

    num_data_points, data_dim = data.shape

    if train_ax_kwargs is None:
        train_ax_kwargs = dict()
    if val_ax_kwargs is None:
        val_ax_kwargs = dict()

    if isinstance(train_ax_kwargs, (list, tuple)):
        assert len(train_ax_kwargs) == data_dim
    if isinstance(val_ax_kwargs, (list, tuple)):
        assert len(val_ax_kwargs) == data_dim

    if isinstance(train_ax_kwargs, dict):
        train_ax_kwargs = [train_ax_kwargs] * data_dim
    if isinstance(val_ax_kwargs, dict):
        val_ax_kwargs = [val_ax_kwargs] * data_dim

    if labels is not None:
        assert len(labels) == data_dim

    artists = []
    for dim in range(data_dim):
        (_a_train,) = ax.plot(
            train_locations,
            data[train_idcs, dim],
            label=labels[dim] if labels is not None else None,
            **train_ax_kwargs[dim],
        )

        (_a_val,) = ax.plot(
            val_locations,
            data[val_idcs, dim],
            **val_ax_kwargs[dim],
        )

        artists.append((_a_train, _a_val))

    return ax, artists


def inject_posterior_mean(
    ax,
    log_dir,
    projmat,
    locations,
    idcs=None,
    smoothing_or_filter="smoothing",
    logfile_suffix="",
    transform_fn=None,
    labels=None,
    mean_ax_kwargs=None,
):
    if transform_fn is None:
        transform_fn = lambda x: x

    if smoothing_or_filter not in ["smoothing", "filter"]:
        raise ValueError(
            f"Received {smoothing_or_filter} for argument smoothing_or_filter."
        )

    _posterior_save_file = (
        log_dir / f"{smoothing_or_filter}_posterior{logfile_suffix}.npz"
    )
    loaded_posterior = np.load(_posterior_save_file)

    vis_dim, state_dim = projmat.shape

    if mean_ax_kwargs is None:
        mean_ax_kwargs = dict()

    if isinstance(mean_ax_kwargs, (list, tuple)):
        assert len(mean_ax_kwargs) == vis_dim

    if isinstance(mean_ax_kwargs, dict):
        mean_ax_kwargs = [mean_ax_kwargs] * vis_dim

    if labels is not None:
        assert len(labels) == vis_dim

    logging.info(
        (
            f"Loaded {smoothing_or_filter} posterior "
            f" :: Shape [T x D] = {loaded_posterior['means'].shape}"
        )
    )

    # projmat.shape            = [D_vis, D_state]
    # posterior["means"].shape = [  T,   D_state]
    means = np.einsum("td,kd->tk", loaded_posterior["means"], projmat)

    if idcs is not None:
        locations = locations[idcs]
        means = means[idcs, :]

    artists = []
    for dim in range(vis_dim):
        (_a,) = ax.plot(
            locations,
            transform_fn(means[:, dim]),
            label=labels[dim] if labels is not None else None,
            **mean_ax_kwargs[dim],
        )
        artists.append(_a)

    return ax, artists


def inject_groundtruth(
    ax,
    gt_data,
    locations,
    idcs=None,
    labels=None,
    gt_ax_kwargs=None,
):
    num_data_points, data_dim = gt_data.shape

    if labels is not None:
        assert len(labels) == data_dim

    if gt_ax_kwargs is None:
        gt_ax_kwargs = dict()

    if isinstance(gt_ax_kwargs, (list, tuple)):
        assert len(gt_ax_kwargs) == data_dim

    if isinstance(gt_ax_kwargs, dict):
        gt_ax_kwargs = [gt_ax_kwargs] * data_dim

    artists = []
    for dim in range(data_dim):
        (_a,) = ax.plot(
            locations[idcs],
            gt_data[idcs, dim],
            label=labels[dim] if labels is not None else None,
            **gt_ax_kwargs[dim],
        )
        artists.append(_a)

    return ax, artists


def inject_posterior_credible_interval(
    ax,
    log_dir,
    projmat,
    locations,
    idcs=None,
    smoothing_or_filter="smoothing",
    logfile_suffix="",
    transform_fn=None,
    labels=None,
    ci_ax_kwargs={},
):
    if transform_fn is None:
        transform_fn = lambda x: x

    if smoothing_or_filter not in ["smoothing", "filter"]:
        raise ValueError(
            f"Received {smoothing_or_filter} for argument smoothing_or_filter."
        )

    _posterior_save_file = (
        log_dir / f"{smoothing_or_filter}_posterior{logfile_suffix}.npz"
    )
    loaded_posterior = np.load(_posterior_save_file)

    vis_dim, state_dim = projmat.shape

    if ci_ax_kwargs is None:
        ci_ax_kwargs = dict()

    if isinstance(ci_ax_kwargs, (list, tuple)):
        assert len(ci_ax_kwargs) == vis_dim

    if isinstance(ci_ax_kwargs, dict):
        ci_ax_kwargs = [ci_ax_kwargs] * vis_dim

    if labels is not None:
        assert len(labels) == vis_dim

    logging.info(
        (
            f"Loaded {smoothing_or_filter} posterior "
            f" :: Shape [T x D] = {loaded_posterior['means'].shape}"
        )
    )

    # projmat.shape            = [D_vis, D_state]
    # posterior["means"].shape = [  T,   D_state]
    marginal_stds = np.stack([np.diag(sig) for sig in loaded_posterior["covs"]])

    means = np.einsum("td,kd->tk", loaded_posterior["means"], projmat)
    marginal_stds = np.sqrt(np.einsum("td,kd->tk", marginal_stds, projmat))

    if idcs is not None:
        locations = np.array(locations[idcs])
        means = means[idcs, :]
        marginal_stds = marginal_stds[idcs, :]

    lower_ci = means - 1.96 * marginal_stds
    upper_ci = means + 1.96 * marginal_stds

    artists = []
    for dim in range(vis_dim):
        _a = ax.fill_between(
            locations,
            y1=transform_fn(lower_ci[:, dim]),
            y2=transform_fn(upper_ci[:, dim]),
            label=labels[dim] if labels is not None else None,
            **ci_ax_kwargs[dim],
        )
        artists.append(_a)

    return ax, artists


def inject_posterior_samples(
    ax,
    log_dir,
    projmat,
    locations,
    idcs=None,
    transform_fn=None,
    labels=None,
    sample_ax_kwargs=None,
):

    if transform_fn is None:
        transform_fn = lambda x: x

    _samples_save_file = log_dir / "posterior_samples.npy"
    loaded_samples = np.load(_samples_save_file)

    num_samples = loaded_samples.shape[0]
    sample_dim, state_dim = projmat.shape

    if sample_ax_kwargs is None:
        sample_ax_kwargs = dict()

    logging.info(
        f"Loaded {num_samples} samples :: Shape [N x T x D] = {loaded_samples.shape}"
    )

    if labels is not None:
        assert len(labels) == sample_dim

    # projmat.shape = [D_vis, D_state]
    # samples.shape = [N, T, D_state]
    samples = np.einsum("ntd,kd->ntk", loaded_samples, projmat)

    if idcs is not None:
        locations = np.array(locations[idcs])
        samples = samples[:, idcs, :]

    for dim in range(sample_dim):
        # Plot one for the legend
        ax.plot(
            locations,
            transform_fn(samples[0, :, dim]),
            label=labels[dim] if labels is not None else None,
            **sample_ax_kwargs,
        )
        for smpl in samples[1:, ...]:
            ax.plot(
                locations,
                transform_fn(smpl[:, dim]),
                **sample_ax_kwargs,
            )

    return ax


def inject_animation_samples(
    ax,
    log_dir,
    sample_idx,
    projmat,
    locations,
    idcs=None,
    transform_fn=None,
    labels=None,
    sample_ax_kwargs={},
):

    if transform_fn is None:
        transform_fn = lambda x: x

    _samples_save_file = log_dir / "animation_samples.npy"
    loaded_samples = np.load(_samples_save_file)

    num_samples = loaded_samples.shape[0]
    sample_dim, state_dim = projmat.shape

    logging.info(
        f"Loaded {num_samples} samples :: Shape [N x T x D] = {loaded_samples.shape}"
    )

    if labels is not None:
        assert len(labels) == sample_dim

    # projmat.shape = [D_vis, D_state]
    # samples.shape = [N, T, D_state]
    samples = np.einsum("ntd,kd->ntk", loaded_samples, projmat)

    if idcs is not None:
        locations = np.array(locations[idcs])
        samples = samples[:, idcs, :]

    for dim in range(sample_dim):
        # Plot one for the legend
        ax.plot(
            locations,
            transform_fn(samples[sample_idx, :, dim]),
            label=labels[dim] if labels is not None else None,
            **sample_ax_kwargs,
        )

    return ax


def inject_dates(
    ax,
    dates,
    labels,
    **ax_kwargs,
):

    ax.vlines(dates, **ax_kwargs)
    for i, d in enumerate(dates):
        ax.annotate(
            labels[i],
            (d, ax_kwargs["ymax"]),
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "C0",
                "pad": 0.2,
                "linewidth": plt.rcParams["xtick.major.width"],
            },
            ha="center",
            va="center",
        )

    return ax


def inject_mcmc_stuff(
    ax_sird,
    ax_beta,
    log_dirs,
    scaling,
    locations,
    sird_ax_kwargs=None,
    beta_ax_kwargs=None,
):

    gamma = 0.06
    eta = 0.002
    L = 25

    # (-400, 900)
    MATERN_FREQS = (
        0.0007692307692307692,
        0.0007692307692307692,
        0.0007692307692307692,
        0.0015384615384615385,
        0.0015384615384615385,
        0.0023076923076923075,
        0.0023076923076923075,
        0.003076923076923077,
        0.003076923076923077,
        0.0038461538461538464,
        0.0038461538461538464,
        0.004615384615384615,
        0.004615384615384615,
        0.005384615384615384,
        0.005384615384615384,
        0.006153846153846154,
        0.006153846153846154,
        0.006923076923076923,
        0.006923076923076923,
        0.007692307692307693,
        0.007692307692307693,
        0.008461538461538461,
        0.008461538461538461,
        0.008461538461538461,
        0.00923076923076923,
        0.00923076923076923,
        0.01,
        0.01,
        0.010769230769230769,
        0.010769230769230769,
        0.011538461538461539,
        0.011538461538461539,
        0.012307692307692308,
        0.012307692307692308,
        0.013076923076923076,
        0.013076923076923076,
        0.013846153846153847,
        0.013846153846153847,
        0.014615384615384615,
        0.014615384615384615,
        0.015384615384615385,
        0.015384615384615385,
        0.016153846153846154,
        0.016153846153846154,
        0.016923076923076923,
        0.016923076923076923,
        0.01769230769230769,
        0.01769230769230769,
        0.01846153846153846,
        0.01846153846153846,
    )

    def superimposed_sinusoidals(x, coeffs):

        # assert len(MATERN_FREQS) == len(MATERN_OFFSETS) == len(coeffs) == len(shifts) == L

        feats = [
            # (((r + c * 1j) * jnp.exp(2 * jnp.pi * freq * 1j * x)).real)
            r * np.cos(2 * np.pi * freq * x) - c * np.sin(2 * np.pi * freq * x)
            for (r, c, freq) in zip(coeffs[::2], coeffs[1::2], MATERN_FREQS[:L])
        ]
        feats = np.stack(feats)
        return np.sum(feats, axis=0).squeeze()

    def calculate_beta_t(
        t,
        coeffs,
    ):

        features = superimposed_sinusoidals(t, coeffs)
        return scipy.special.expit(features + scipy.special.logit(0.1)).squeeze()

    def dz_dt(
        t,
        z,
        coeffs,
        population,
    ):
        """
        SIRD
        """
        s = z[0]
        i = z[1]
        r = z[2]
        d = z[3]

        beta_t = calculate_beta_t(
            t,
            coeffs,
        )

        ds_dt = (-beta_t * s * i) / population
        di_dt = (beta_t * s * i) / population - gamma * i - eta * i
        dr_dt = gamma * i
        dd_dt = eta * i
        return np.stack([ds_dt, di_dt, dr_dt, dd_dt])

    def solve_sird(theta):
        return solve_ivp(
            functools.partial(dz_dt, coeffs=theta, population=83783945.0),
            (0.0, locations.shape[0]),
            y0=np.array([83783945.0, 1e-2, 0.0, 0.0]),
            t_eval=np.arange(locations.shape[0]),
        ).y.T

    if sird_ax_kwargs is None:
        sird_ax_kwargs = {"mean": {}, "ci": {}, "samples": {}}

    if beta_ax_kwargs is None:
        beta_ax_kwargs = {"mean": {}, "ci": {}, "samples": {}}

    result_dicts = [np.load(l / "result.npz") for l in log_dirs]

    merged_betas = np.concatenate([d["sampled_beta"] for d in result_dicts], axis=0)
    # merged_states = np.concatenate([d["sampled_sird"] for d in result_dicts], axis=0)

    merged_coeffs = np.concatenate([d["sampled_coeffs"] for d in result_dicts], axis=0)

    logging.info(f"Found {merged_coeffs.shape[0]} samples.")

    merged_states = (
        np.stack([solve_sird(coeffs)[:, 1] for coeffs in merged_coeffs], axis=0)
    ) * scaling

    print(merged_states.shape)

    smpl_idcs = np.random.choice(merged_states.shape[0], size=3)

    samples_states = merged_states[smpl_idcs, ...]
    samples_beta = merged_betas[smpl_idcs, ...]

    mean_beta = np.mean(merged_betas, 0)
    mean_states = np.mean(merged_states, 0)

    ci_beta = np.percentile(merged_betas, (2.5, 97.5), axis=0)
    ci_states = np.percentile(merged_states, (2.5, 97.5), axis=0)

    (state_handle,) = ax_sird.plot(locations, mean_states, **sird_ax_kwargs["mean"])
    ax_sird.fill_between(
        locations, ci_states[0, ...], ci_states[1, ...], **sird_ax_kwargs["ci"]
    )

    (beta_handle,) = ax_beta.plot(locations, mean_beta, **beta_ax_kwargs["mean"])
    ax_beta.fill_between(
        locations, ci_beta[0, ...], ci_beta[1, ...], **beta_ax_kwargs["ci"]
    )

    for s in range(3):
        ax_sird.plot(locations, samples_states[s], **sird_ax_kwargs["samples"])
        ax_beta.plot(locations, samples_beta[s], **beta_ax_kwargs["samples"])

    return ax_sird, ax_beta, (state_handle, beta_handle)
