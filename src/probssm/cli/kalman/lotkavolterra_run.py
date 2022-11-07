import argparse
import logging
import pathlib
from datetime import datetime

import numpy as np
import probnum as pn
import scipy.linalg
import scipy.special
from probnum import filtsmooth, problems, randprocs, randvars

import probssm
from probssm.util import find_nearest, unions1d

from ._likelihoods import LVLikelihood

# from probnumeval import timeseries as pn_eval_timeseries
# from probnumeval import utils as pn_eval_utils



logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - log-sird-vacc - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument("--logdir", type=str, default="./run")
    parser.add_argument("--logdir-suffix", type=str, required=False)
    parser.add_argument("--loaddir", type=str, required=False)

    data_arg_group = parser.add_argument_group(title="Data")
    data_arg_group.add_argument("--num-train", type=int, required=False)
    data_arg_group.add_argument("--num-data-points", type=int, default=100)

    model_arg_group = parser.add_argument_group(title="Model Hyperparameters")
    model_arg_group.add_argument("--data-measurement-cov", type=float, default=1.0)
    model_arg_group.add_argument("--ode-measurement-cov", type=float, default=0.1)

    _default_filter_stepsize = 1.0 / 100.0
    model_arg_group.add_argument(
        "--filter-step-size", type=float, default=_default_filter_stepsize
    )
    model_arg_group.add_argument("--purely-mechanistic", action="store_true")
    model_arg_group.add_argument("--purely-data", action="store_true")

    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--pn-forward-implementation", type=str, default="sqrt")
    parser.add_argument("--pn-backward-implementation", type=str, default="sqrt")

    x_process_arg_group = parser.add_argument_group(title="X-process")
    x_process_arg_group.add_argument("--x-process-diffusion", type=float, default=1.0)
    x_process_arg_group.add_argument("--x-process-ordint", type=int, default=2)

    param_processes = parser.add_argument_group(title="param-processes")
    param_processes.add_argument("--alpha-prior-mean", type=float, default=0.4)
    param_processes.add_argument("--alpha-process-diffusion", type=float, default=0.01)
    param_processes.add_argument("--alpha-process-ordint", type=int, default=0)
    param_processes.add_argument(
        "--alpha-process-lengthscale", type=float, default=14.0
    )

    param_processes.add_argument("--beta-prior-mean", type=float, default=0.05)
    param_processes.add_argument("--beta-process-diffusion", type=float, default=0.01)
    param_processes.add_argument("--beta-process-ordint", type=int, default=0)
    param_processes.add_argument("--beta-process-lengthscale", type=float, default=14.0)

    param_processes.add_argument("--gamma-prior-mean", type=float, default=0.4)
    param_processes.add_argument("--gamma-process-diffusion", type=float, default=0.01)
    param_processes.add_argument("--gamma-process-ordint", type=int, default=0)
    param_processes.add_argument(
        "--gamma-process-lengthscale", type=float, default=14.0
    )

    param_processes.add_argument("--delta-prior-mean", type=float, default=0.05)
    param_processes.add_argument("--delta-process-diffusion", type=float, default=0.01)
    param_processes.add_argument("--delta-process-ordint", type=int, default=0)
    param_processes.add_argument(
        "--delta-process-lengthscale", type=float, default=14.0
    )

    # Checks
    arg_namespace = parser.parse_args()
    if arg_namespace.purely_mechanistic and arg_namespace.purely_data:
        raise ValueError("Can only set --purely-mechanistic XOR --purely-data.")

    return arg_namespace


def main():
    args = parse_args()

    logging.info("===== Starting Log SIRD data experiment with vaccination data =====")

    rng = np.random.default_rng(seed=123)

    STATE_DIM = 2
    OBSERVATION_DIM = 2

    X_PROCESS_NUM = 0
    ALPHA_PROCESS_NUM = 1
    BETA_PROCESS_NUM = 2
    GAMMA_PROCESS_NUM = 3
    DELTA_PROCESS_NUM = 4

    # ##################################################################################
    # PRIOR
    # ##################################################################################

    forward_implementation = args.pn_forward_implementation
    backward_implementation = args.pn_backward_implementation

    ode_transition = pn.randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=args.x_process_ordint,
        wiener_process_dimension=STATE_DIM,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    ode_transition._dispersion_matrix = (
        ode_transition._dispersion_matrix * args.x_process_diffusion
    )

    alpha_transition = pn.randprocs.markov.integrator.MaternTransition(
        num_derivatives=args.alpha_process_ordint,
        wiener_process_dimension=1,
        lengthscale=args.beta_process_lengthscale,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    alpha_transition._dispersion_matrix = (
        alpha_transition._dispersion_matrix * args.beta_process_diffusion
    )

    beta_transition = pn.randprocs.markov.integrator.MaternTransition(
        num_derivatives=args.beta_process_ordint,
        wiener_process_dimension=1,
        lengthscale=args.beta_process_lengthscale,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    beta_transition._dispersion_matrix = (
        beta_transition._dispersion_matrix * args.beta_process_diffusion
    )

    gamma_transition = pn.randprocs.markov.integrator.MaternTransition(
        num_derivatives=args.gamma_process_ordint,
        wiener_process_dimension=1,
        lengthscale=args.gamma_process_lengthscale,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    gamma_transition._dispersion_matrix = (
        gamma_transition._dispersion_matrix * args.gamma_process_diffusion
    )

    delta_transition = pn.randprocs.markov.integrator.MaternTransition(
        num_derivatives=args.delta_process_ordint,
        wiener_process_dimension=1,
        lengthscale=args.delta_process_lengthscale,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    delta_transition._dispersion_matrix = (
        delta_transition._dispersion_matrix * args.delta_process_diffusion
    )

    prior_transition = probssm.stacked_ssm.StackedTransition(
        transitions=(
            ode_transition,
            alpha_transition,
            beta_transition,
            gamma_transition,
            delta_transition,
        ),
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    # Set up initial conditions

    # ##################################################################################
    # ODE LIKELIHOOD
    # ##################################################################################

    alpha_offset = -np.log(args.alpha_prior_mean)
    beta_offset = -np.log(args.beta_prior_mean)
    gamma_offset = -np.log(args.gamma_prior_mean)
    delta_offset = -np.log(args.delta_prior_mean)
    alpha_link_fn = lambda x: np.exp(x - alpha_offset)
    beta_link_fn = lambda x: np.exp(x - beta_offset)
    gamma_link_fn = lambda x: np.exp(x - gamma_offset)
    delta_link_fn = lambda x: np.exp(x - delta_offset)

    alpha_link_fn_deriv = alpha_link_fn
    beta_link_fn_deriv = beta_link_fn
    gamma_link_fn_deriv = gamma_link_fn
    delta_link_fn_deriv = delta_link_fn

    alpha_inverse_link_fn = lambda x: np.log(x) + alpha_offset
    beta_inverse_link_fn = lambda x: np.log(x) + beta_offset
    gamma_inverse_link_fn = lambda x: np.log(x) + gamma_offset
    delta_inverse_link_fn = lambda x: np.log(x) + delta_offset

    assert np.isclose(
        alpha_link_fn(alpha_inverse_link_fn(args.alpha_prior_mean)),
        args.alpha_prior_mean,
    )
    assert np.isclose(alpha_link_fn(0.0), args.alpha_prior_mean)

    assert np.isclose(
        beta_link_fn(beta_inverse_link_fn(args.beta_prior_mean)),
        args.beta_prior_mean,
    )
    assert np.isclose(beta_link_fn(0.0), args.beta_prior_mean)

    assert np.isclose(
        gamma_link_fn(gamma_inverse_link_fn(args.gamma_prior_mean)),
        args.gamma_prior_mean,
    )
    assert np.isclose(gamma_link_fn(0.0), args.gamma_prior_mean)

    assert np.isclose(
        delta_link_fn(delta_inverse_link_fn(args.delta_prior_mean)),
        args.delta_prior_mean,
    )
    assert np.isclose(delta_link_fn(0.0), args.delta_prior_mean)

    ode_likelihood = LVLikelihood(
        prior=prior_transition,
        alpha_link_fn=alpha_link_fn,
        alpha_link_fn_deriv=alpha_link_fn_deriv,
        beta_link_fn=beta_link_fn,
        beta_link_fn_deriv=beta_link_fn_deriv,
        gamma_link_fn=gamma_link_fn,
        gamma_link_fn_deriv=gamma_link_fn_deriv,
        delta_link_fn=delta_link_fn,
        delta_link_fn_deriv=delta_link_fn_deriv,
    )

    process_idcs = prior_transition.state_idcs

    u_0, v_0 = 3.0, 3.0

    # Mean
    lv0 = np.array([u_0, v_0])
    vel0 = probssm.ivp.lv_rhs(
        0.0,
        lv0,
        alpha=args.alpha_prior_mean,
        beta=args.beta_prior_mean,
        gamma=args.gamma_prior_mean,
        delta=args.delta_prior_mean,
    )

    logging.debug(f"Initial SIRD mean: {lv0}")

    init_mean = np.zeros((prior_transition.state_dimension,))

    init_mean[process_idcs[X_PROCESS_NUM]["state_d0"]] = lv0
    init_mean[process_idcs[X_PROCESS_NUM]["state_d1"]] = vel0
    init_mean[process_idcs[X_PROCESS_NUM]["state_d2"]] = 1e-3

    # Cov

    init_marginal_vars = 1e-1 * np.ones((prior_transition.state_dimension,))
    init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d0"]] = np.array([1e-4, 1e-4])

    init_marginal_vars[process_idcs[ALPHA_PROCESS_NUM]["state_d0"]] = 0.1
    init_marginal_vars[process_idcs[BETA_PROCESS_NUM]["state_d0"]] = 0.01
    init_marginal_vars[process_idcs[GAMMA_PROCESS_NUM]["state_d0"]] = 0.1
    init_marginal_vars[process_idcs[DELTA_PROCESS_NUM]["state_d0"]] = 0.01
    init_marginal_vars[process_idcs[ALPHA_PROCESS_NUM]["state_d1"]] = 0.001
    init_marginal_vars[process_idcs[BETA_PROCESS_NUM]["state_d1"]] = 0.001
    init_marginal_vars[process_idcs[GAMMA_PROCESS_NUM]["state_d1"]] = 0.001
    init_marginal_vars[process_idcs[DELTA_PROCESS_NUM]["state_d1"]] = 0.001

    init_cov = np.diag(init_marginal_vars)

    initrv = randvars.Normal(init_mean, init_cov)

    time_domain = (0.0, 60.0)
    prior_process = randprocs.markov.MarkovProcess(
        transition=prior_transition, initrv=initrv, initarg=time_domain[0]
    )

    # Check jacobians

    _point = (
        prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0)
        @ prior_transition.proj2process(X_PROCESS_NUM)
        @ initrv.mean
    )
    _alpha = np.array(1.0)
    _beta = np.array(0.05)
    _gamma = np.array(1.0)
    _delta = np.array(0.05)
    _t = 0.1
    _m = initrv.mean

    ode_likelihood.check_jacobians(_t, _point, _alpha, _beta, _gamma, _delta, _m)

    # ##################################################################################
    # BUILD MODEL
    # ##################################################################################

    # ODE measurements
    measurement_matrix_ode = args.ode_measurement_cov * np.eye(STATE_DIM)
    measurement_noiserv_ode = randvars.Normal(mean=np.zeros(STATE_DIM), cov=measurement_matrix_ode)
    measurement_model_ode = randprocs.markov.discrete.NonlinearGaussian(
        input_dim=initrv.mean.size,
        output_dim=STATE_DIM,
        transition_fun=ode_likelihood.measure_ode,
        noise_fun=lambda t: measurement_noiserv_ode,
        transition_fun_jacobian=ode_likelihood.measure_ode_jacobian,
    )

    # EKF
    linearized_measurement_model_ode = filtsmooth.gaussian.approx.DiscreteEKFComponent(
        non_linear_model=measurement_model_ode,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    # Data measurements
    measurement_matrix_data = args.data_measurement_cov * np.eye(OBSERVATION_DIM)

    proj_state_to_UV = prior_transition.proj2coord(
        proc=X_PROCESS_NUM, coord=0
    ) @ prior_transition.proj2process(X_PROCESS_NUM)

    measurement_noiserv_data = randvars.Normal(mean=np.zeros(OBSERVATION_DIM), cov=measurement_matrix_data)
    measurement_model_data = randprocs.markov.discrete.LTIGaussian(
        transition_matrix=proj_state_to_UV,
        noise=measurement_noiserv_data,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    # ##################################################################################
    # DATA
    # ##################################################################################

    num_data_points = args.num_data_points
    # Split into train and validation set
    if args.num_train is None:
        num_train = num_data_points
    elif args.num_train < 0:
        num_train = num_data_points + args.num_train
    else:
        num_train = args.num_train

    test_set_conditions = [
        lambda i: i > num_train,
        # lambda i: i % 2 != 0,
    ]

    train_idcs, val_idcs = probssm.split_train_test(
        all_idcs=np.arange(num_data_points, dtype=int),
        list_lambda_predicates_test=test_set_conditions,
    )

    # Generate data

    data_grid = np.linspace(
        time_domain[0] + 0.5,
        time_domain[1] - 0.5,
        num=args.num_data_points,
        endpoint=True,
    )
    ode_grid = np.arange(
        *time_domain,
        step=args.filter_step_size,
    )

    dense_grid = unions1d(data_grid, ode_grid)

    if args.loaddir is None:

        # Sample entire latent state from prior
        prior_samples = prior_process.sample(rng=rng, args=dense_grid)

        sampled_prior_alpha_t = alpha_link_fn(
            (
                prior_transition.proj2coord(proc=ALPHA_PROCESS_NUM, coord=0)
                @ prior_transition.proj2process(ALPHA_PROCESS_NUM)
                @ prior_samples.T
            ).T
        )
        sampled_prior_beta_t = beta_link_fn(
            (
                prior_transition.proj2coord(proc=BETA_PROCESS_NUM, coord=0)
                @ prior_transition.proj2process(BETA_PROCESS_NUM)
                @ prior_samples.T
            ).T
        )
        sampled_prior_gamma_t = gamma_link_fn(
            (
                prior_transition.proj2coord(proc=GAMMA_PROCESS_NUM, coord=0)
                @ prior_transition.proj2process(GAMMA_PROCESS_NUM)
                @ prior_samples.T
            ).T
        )
        sampled_prior_delta_t = delta_link_fn(
            (
                prior_transition.proj2coord(proc=DELTA_PROCESS_NUM, coord=0)
                @ prior_transition.proj2process(DELTA_PROCESS_NUM)
                @ prior_samples.T
            ).T
        )

        # Compute "groundtruth" solution

        # Set up vector field function that uses the sampled (prior) parameters
        def lv_with_sampled_params(t, state):
            idx, _ = find_nearest(dense_grid, t)
            alpha = sampled_prior_alpha_t[idx].squeeze()
            beta = sampled_prior_beta_t[idx].squeeze()
            gamma = sampled_prior_gamma_t[idx].squeeze()
            delta = sampled_prior_delta_t[idx].squeeze()
            return ode_likelihood.rhs(
                t, state, alpha=alpha, beta=beta, gamma=gamma, delta=delta
            )

        # Solve the LV initial value problem
        lv_solution = scipy.integrate.solve_ivp(
            fun=lv_with_sampled_params,
            t_span=time_domain,
            dense_output=True,
            y0=lv0,
            method="LSODA",
        )

        # Interpolate to obtain ODE solution groundtruth at regular grid
        data_sol = lv_solution.sol(data_grid).T
        gt_sol = lv_solution.sol(dense_grid).T

        _datalist = []
        for s in data_sol:
            s_noisy = s + np.random.multivariate_normal(
                np.zeros_like(s), args.data_measurement_cov * np.eye(s.shape[0])
            )
            _datalist.append(s_noisy)

        data = np.stack(_datalist)

    else:
        loaddir = pathlib.Path(args.loaddir)
        loaded = np.load(loaddir / "data_info.npz")
        data = loaded["data"]

        gt_sol = loaded["gt_data"]
        sampled_prior_alpha_t = loaded["gt_alpha"]
        sampled_prior_beta_t = loaded["gt_beta"]
        sampled_prior_gamma_t = loaded["gt_gamma"]
        sampled_prior_delta_t = loaded["gt_delta"]

    # Pseudo observations for ODE measurements
    zero_data = np.zeros(STATE_DIM, dtype=np.float64)

    # ##################################################################################
    # Run algorithm
    # ##################################################################################

    # Set up log dir
    if args.logdir_suffix is None:
        logdir_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        logdir_suffix = args.logdir_suffix
    log_dir = pathlib.Path(f"{args.logdir}_{logdir_suffix}").absolute()
    log_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Logging to {log_dir}")

    logging.info(f"Solving on time domain: t in {time_domain}")

    merged_locations = unions1d(data_grid, ode_grid)

    merged_observations = []
    merged_measmods = []

    data_idx = 0
    ode_idx = 0

    for loc in merged_locations:
        if np.in1d(loc, data_grid):
            merged_observations.append(data[data_idx, :])
            merged_measmods.append(measurement_model_data)
            data_idx += 1

        elif np.in1d(loc, ode_grid):
            merged_observations.append(np.array(zero_data))
            merged_measmods.append(linearized_measurement_model_ode)
            ode_idx += 1
        else:
            pass

    print(f"{data_idx} / {merged_locations.size} data locations")
    print(f"{ode_idx} / {merged_locations.size} ODE locations")

    merged_regression_problem = problems.TimeSeriesRegressionProblem(
        observations=merged_observations,
        locations=merged_locations,
        measurement_models=merged_measmods,
    )

    assert len(merged_observations) == len(merged_measmods) == len(merged_locations)

    kalman_filter = filtsmooth.gaussian.Kalman(prior_process)

    posterior_and_map = []

    for i, (posterior, _) in enumerate(
        kalman_filter.iterated_filtsmooth_posterior_generator(
            regression_problem=merged_regression_problem
        )
    ):
        print(f"Posterior nr. {i}")
        if i == 0:
            posterior_and_map.append(posterior)

        # Manually stop after x iterations
        if i > 20:
            break

    posterior_and_map.append(posterior)

    _posterior_save_file = log_dir / "smoothing_posterior.npz"
    _map_posterior_save_file = log_dir / "smoothing_posterior_map.npz"

    posterior, map_posterior = posterior_and_map

    np.savez(
        _posterior_save_file,
        means=np.stack([s.mean for s in posterior]),
        covs=np.stack([s.cov for s in posterior]),
    )
    np.savez(
        _map_posterior_save_file,
        means=np.stack([s.mean for s in map_posterior]),
        covs=np.stack([s.cov for s in map_posterior]),
    )

    if args.num_samples is not None and args.num_samples > 0:
        logging.info(f"Drawing {args.num_samples} samples from posterior.")

        samples = posterior.sample(rng=rng, size=args.num_samples)

        _samples_save_file = log_dir / "posterior_samples.npy"
        np.save(_samples_save_file, samples)

        logging.info(f"Saved posterior samples to {_samples_save_file}.")

    logging.info("Computation done. Finalize")

    def param_approx(locs, _map=False):
        post = map_posterior if _map else posterior
        proj2procs = np.vstack(
            [
                prior_transition.proj2process(ALPHA_PROCESS_NUM),
                prior_transition.proj2process(BETA_PROCESS_NUM),
                prior_transition.proj2process(GAMMA_PROCESS_NUM),
                prior_transition.proj2process(DELTA_PROCESS_NUM),
            ]
        )
        proj2coords = scipy.linalg.block_diag(
            prior_transition.proj2coord(proc=ALPHA_PROCESS_NUM, coord=0),
            prior_transition.proj2coord(proc=BETA_PROCESS_NUM, coord=0),
            prior_transition.proj2coord(proc=GAMMA_PROCESS_NUM, coord=0),
            prior_transition.proj2coord(proc=DELTA_PROCESS_NUM, coord=0),
        )

        param_rvs = [(proj2coords @ proj2procs @ rv) for rv in post(locs)]
        return param_rvs

    # ALL PARAMS

    param_approx_first_logspace = param_approx(dense_grid, _map=False)
    param_approx_map_logspace = param_approx(dense_grid, _map=True)

    param_approx_first_mean_logspace = np.stack(
        [r.mean for r in param_approx_first_logspace]
    )
    param_approx_map_mean_logspace = np.stack(
        [r.mean for r in param_approx_map_logspace]
    )

    param_approx_first_mean_linspace = np.stack(
        [
            np.stack(
                [
                    lf(mu)
                    for lf, mu in zip(
                        (
                            alpha_link_fn,
                            beta_link_fn,
                            gamma_link_fn,
                            delta_link_fn,
                        ),
                        r.mean,
                    )
                ]
            )
            for r in param_approx_first_logspace
        ]
    )
    param_approx_map_mean_linspace = np.stack(
        [
            np.stack(
                [
                    lf(mu)
                    for lf, mu in zip(
                        (
                            alpha_link_fn,
                            beta_link_fn,
                            gamma_link_fn,
                            delta_link_fn,
                        ),
                        r.mean,
                    )
                ]
            )
            for r in param_approx_map_logspace
        ]
    )

    stacked_sampled_params_logspace = np.concatenate(
        [
            alpha_inverse_link_fn(sampled_prior_alpha_t),
            beta_inverse_link_fn(sampled_prior_beta_t),
            gamma_inverse_link_fn(sampled_prior_gamma_t),
            delta_inverse_link_fn(sampled_prior_delta_t),
        ],
        -1,
    )

    stacked_sampled_params_linspace = np.concatenate(
        [
            sampled_prior_alpha_t,
            sampled_prior_beta_t,
            sampled_prior_gamma_t,
            sampled_prior_delta_t,
        ],
        -1,
    )

    # print(
    #     f"Chi2 95 percentile: {pn_eval_utils.chi2_confidence_intervals(4, perc=0.95)}"
    # )

    # # CHI2 in Log space ----------------------------------------------------------------
    # # First
    # chi2_param_first_logspace = pn_eval_timeseries.anees(
    #     approximate_solution=lambda x: param_approx_first_logspace,
    #     reference_solution=lambda x: stacked_sampled_params_logspace,
    #     locations=dense_grid,
    # )
    # logging.info(f"ANEES statistic first (logspace): {chi2_param_first_logspace}")

    # # MAP
    # chi2_param_map_logspace = pn_eval_timeseries.anees(
    #     approximate_solution=lambda x: param_approx_map_logspace,
    #     reference_solution=lambda x: stacked_sampled_params_logspace,
    #     locations=dense_grid,
    # )
    # logging.info(f"ANEES statistic MAP (logspace): {chi2_param_map_logspace}")

    # # RMSE in Log space ----------------------------------------------------------------
    # # First

    # rmse_param_first_logspace = np.linalg.norm(
    #     param_approx_first_mean_logspace - stacked_sampled_params_logspace
    # ) / np.sqrt(sampled_prior_beta_t.size)

    # logging.info(f"RMSE first (logspace): {rmse_param_first_logspace}")

    # # MAP
    # rmse_param_map_logspace = np.linalg.norm(
    #     param_approx_map_mean_logspace - stacked_sampled_params_logspace
    # ) / np.sqrt(sampled_prior_beta_t.size)
    # logging.info(f"RMSE MAP (logspace): {rmse_param_map_logspace}")

    # # RMSE in Lin space ----------------------------------------------------------------
    # # First

    # rmse_param_first_linspace = np.linalg.norm(
    #     param_approx_first_mean_linspace - stacked_sampled_params_linspace
    # ) / np.sqrt(sampled_prior_beta_t.size)

    # logging.info(f"RMSE first (linspace): {rmse_param_first_linspace}")

    # # MAP
    # rmse_param_map_linspace = np.linalg.norm(
    #     param_approx_map_mean_linspace - stacked_sampled_params_linspace
    # ) / np.sqrt(sampled_prior_beta_t.size)
    # logging.info(f"RMSE MAP (linspace): {rmse_param_map_linspace}")

    # ##################################################################################
    # Finalize
    # ##################################################################################
    projections_dict = {
        "E_x": prior_transition.proj2process(X_PROCESS_NUM),
        "E_alpha": prior_transition.proj2process(ALPHA_PROCESS_NUM),
        "E_beta": prior_transition.proj2process(BETA_PROCESS_NUM),
        "E_gamma": prior_transition.proj2process(GAMMA_PROCESS_NUM),
        "E_delta": prior_transition.proj2process(DELTA_PROCESS_NUM),
        "E0_x": prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0),
        "E0_alpha": prior_transition.proj2coord(proc=ALPHA_PROCESS_NUM, coord=0),
        "E0_beta": prior_transition.proj2coord(proc=BETA_PROCESS_NUM, coord=0),
        "E0_gamma": prior_transition.proj2coord(proc=GAMMA_PROCESS_NUM, coord=0),
        "E0_delta": prior_transition.proj2coord(proc=DELTA_PROCESS_NUM, coord=0),
    }
    _projections_save_file = log_dir / "projections.npz"
    np.savez(_projections_save_file, **projections_dict)
    logging.info(f"Saved projections matrices to {_projections_save_file}.")

    data_dict = {
        "data": data,
        "gt_data": gt_sol,
        "gt_alpha": sampled_prior_alpha_t,
        "gt_beta": sampled_prior_beta_t,
        "gt_gamma": sampled_prior_gamma_t,
        "gt_delta": sampled_prior_delta_t,
        "time_domain": np.array(time_domain),
        "dense_grid": dense_grid,
        "data_grid": data_grid,
        "ode_grid": ode_grid,
        "train_idcs": train_idcs,
        "val_idcs": val_idcs,
    }
    _data_info_save_file = log_dir / "data_info.npz"
    np.savez(_data_info_save_file, **data_dict)
    logging.info(f"Saved data info to {_data_info_save_file}.")

    # info = {
    #     "chi2_param_first_logspace": chi2_param_first_logspace,
    #     "chi2_param_map_logspace": chi2_param_map_logspace,
    #     "rmse_param_first_logspace": rmse_param_first_logspace,
    #     "rmse_param_map_logspace": rmse_param_map_logspace,
    #     "rmse_param_first_linspace": rmse_param_first_linspace,
    #     "rmse_param_map_linspace": rmse_param_map_linspace,
    # }
    _args_save_file = log_dir / "info.json"
    probssm.args_to_json(_args_save_file, args=args, kwargs={})
    logging.info(f"Saved info dict to {_args_save_file}.")


if __name__ == "__main__":
    main()
