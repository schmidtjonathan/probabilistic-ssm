import argparse
import functools
import logging
import pathlib
import time
from datetime import datetime

import numpy as np
import probnum as pn
import scipy.linalg
import scipy.special
from probnum import filtsmooth, problems, randprocs, randvars

import probssm

from ._likelihoods import LogSIRDLikelihood

logging.basicConfig(
    level=logging.INFO,
    format=">>> %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument("--logdir", type=str, default="./run")
    parser.add_argument("--logdir-suffix", type=str, required=False)

    data_arg_group = parser.add_argument_group(title="Data")
    data_arg_group.add_argument("--country", type=str, default="Germany")
    data_arg_group.add_argument(
        "--scaling", type=str, choices=["none", "cpt"], default="cpt"
    )
    data_arg_group.add_argument("--num-train", type=int, required=False)
    data_arg_group.add_argument("--num-extrapolate", type=int, default=0)

    model_arg_group = parser.add_argument_group(title="Model Hyperparameters")
    model_arg_group.add_argument("--sigmoid-slope", type=float, default=0.01)
    model_arg_group.add_argument(
        "--gamma", type=float, default=0.06, help="Recovery rate"
    )
    model_arg_group.add_argument(
        "--eta", type=float, default=0.002, help="Mortality rate"
    )

    model_arg_group.add_argument("--data-measurement-cov", type=float, default=0.01)
    model_arg_group.add_argument("--ode-measurement-cov", type=float, default=0.0)

    _default_filter_stepsize = 1.0 / 24.0
    model_arg_group.add_argument(
        "--filter-step-size", type=float, default=_default_filter_stepsize
    )
    model_arg_group.add_argument("--purely-mechanistic", action="store_true")
    model_arg_group.add_argument("--purely-data", action="store_true")

    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--save-intermediate-filtering", type=int, required=False)
    parser.add_argument("--pn-forward-implementation", type=str, default="sqrt")
    parser.add_argument("--pn-backward-implementation", type=str, default="sqrt")

    x_process_arg_group = parser.add_argument_group(title="X-process")
    x_process_arg_group.add_argument("--x-process-diffusion", type=float, default=1.0)
    x_process_arg_group.add_argument("--x-process-ordint", type=int, default=2)

    beta_process_arg_group = parser.add_argument_group(title="beta-process")
    beta_process_arg_group.add_argument("--beta-prior-mean", type=float, default=0.0)
    beta_process_arg_group.add_argument(
        "--beta-process-diffusion", type=float, default=1.0
    )
    beta_process_arg_group.add_argument("--beta-process-ordint", type=int, default=0)
    beta_process_arg_group.add_argument(
        "--beta-process-lengthscale", type=float, default=1.0
    )

    # Checks
    arg_namespace = parser.parse_args()
    if arg_namespace.purely_mechanistic and arg_namespace.purely_data:
        raise ValueError("Can only set --purely-mechanistic XOR --purely-data.")

    return arg_namespace


def main():
    args = parse_args()

    logging.info("===== Starting Log SIRD data experiment =====")

    rng = np.random.default_rng(seed=123)

    STATE_DIM = 4
    OBSERVATION_DIM = 3

    X_PROCESS_NUM = 0
    BETA_PROCESS_NUM = 1

    # Set up log dir
    if args.logdir_suffix is None:
        logdir_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        logdir_suffix = args.logdir_suffix
    log_dir = pathlib.Path(f"{args.logdir}_{logdir_suffix}").absolute()
    log_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Logging to {log_dir}")

    # ##################################################################################
    # DATA
    # ##################################################################################

    # COVID-data
    day_zero, date_range_x, SIRD_data, population = probssm.data.load_COVID_data(
        country="Germany", num_data_points=556
    )

    num_covid_data_points = SIRD_data.shape[0]

    if args.scaling == "cpt":
        cases_per_thousand_scaling = 1e3 / population

        population = population * cases_per_thousand_scaling
        assert np.isclose(population, 1000.0), population

        SIRD_data = SIRD_data * cases_per_thousand_scaling

    # Transform data to log
    SIRD_data += 1e-5
    SIRD_data = np.log(SIRD_data)

    # Pseudo observations for ODE measurements
    zero_data = np.zeros(STATE_DIM, dtype=np.float64)

    # Split into train and validation set
    if args.num_train is None:
        num_train = num_covid_data_points
    elif args.num_train < 0:
        num_train = num_covid_data_points + args.num_train
    else:
        num_train = args.num_train

    test_set_conditions = [
        lambda i: i >= num_train,
        # lambda i: i % 2 != 0,
    ]

    train_idcs, val_idcs = probssm.split_train_test(
        all_idcs=np.arange(num_covid_data_points, dtype=int),
        list_lambda_predicates_test=test_set_conditions,
    )

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

    lf_transition = pn.randprocs.markov.integrator.MaternTransition(
        num_derivatives=args.beta_process_ordint,
        wiener_process_dimension=1,
        lengthscale=args.beta_process_lengthscale,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    lf_transition._dispersion_matrix = (
        lf_transition._dispersion_matrix * args.beta_process_diffusion
    )

    prior_transition = probssm.stacked_ssm.StackedTransition(
        transitions=(ode_transition, lf_transition),
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    # Set up initial conditions

    # ##################################################################################
    # ODE LIKELIHOOD
    # ##################################################################################

    # Link functions

    sigmoid_x_offset = -scipy.special.logit(args.beta_prior_mean)
    beta_link_fn = functools.partial(
        probssm.util.sloped_sigmoid,
        slope=args.sigmoid_slope,
        x_offset=sigmoid_x_offset,
    )
    beta_link_fn_deriv = functools.partial(
        probssm.util.d_sloped_sigmoid,
        slope=args.sigmoid_slope,
        x_offset=sigmoid_x_offset,
    )
    beta_inverse_link_fn = (
        lambda x: (scipy.special.logit(x) + sigmoid_x_offset) / args.sigmoid_slope
    )

    assert np.isclose(
        beta_link_fn(beta_inverse_link_fn(args.beta_prior_mean)), args.beta_prior_mean
    )
    assert np.isclose(beta_link_fn(0.0), args.beta_prior_mean)

    ode_parameters = {
        "gamma": args.gamma,
        "eta": args.eta,
        "population_count": population,
    }

    ode_likelihood = LogSIRDLikelihood(
        prior=prior_transition,
        ode_parameters=ode_parameters,
        beta_link_fn=beta_link_fn,
        beta_link_fn_deriv=beta_link_fn_deriv,
    )

    process_idcs = prior_transition.state_idcs

    # Mean
    sird_0 = np.array(SIRD_data[0, :4])
    init_sird_vel = 1e-3
    init_beta_vel = 0.0

    logging.debug(f"Initial SIRD mean: {np.exp(sird_0)}")

    init_mean = np.zeros((prior_transition.state_dimension,))

    init_mean[process_idcs[X_PROCESS_NUM]["state_d0"]] = sird_0
    init_mean[process_idcs[X_PROCESS_NUM]["state_d1"]] = init_sird_vel
    init_mean[process_idcs[X_PROCESS_NUM]["state_d2"]] = init_sird_vel

    # Set to inverse of link function
    init_mean[process_idcs[BETA_PROCESS_NUM]["state_d0"]] = beta_inverse_link_fn(
        args.beta_prior_mean
    )

    init_mean[process_idcs[BETA_PROCESS_NUM]["state_d1"]] = init_beta_vel

    # Cov
    sigma_sird = 0.001 * np.ones_like(sird_0)

    # Initialize the beta process at its stationary covariance
    stationary_beta_cov = scipy.linalg.solve_continuous_lyapunov(
        lf_transition.drift_matrix,
        -(lf_transition.dispersion_matrix @ lf_transition.dispersion_matrix.T),
    )
    sigma_beta = stationary_beta_cov[0, 0]
    sigma_velocity = 0.001

    init_marginal_vars = 1e-7 * np.ones((prior_transition.state_dimension,))
    init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d0"]] = sigma_sird
    init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d1"]] = sigma_velocity
    init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d2"]] = sigma_velocity

    init_marginal_vars[process_idcs[BETA_PROCESS_NUM]["state_d0"]] = sigma_beta
    init_marginal_vars[process_idcs[BETA_PROCESS_NUM]["state_d1"]] = sigma_velocity

    init_cov = np.diag(init_marginal_vars)

    initrv = randvars.Normal(init_mean, init_cov)

    time_domain = (0.0, float(num_covid_data_points + args.num_extrapolate))
    prior_process = randprocs.markov.MarkovProcess(
        transition=prior_transition, initrv=initrv, initarg=time_domain[0]
    )

    # Check jacobians

    _point = (
        prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0)
        @ prior_transition.proj2process(X_PROCESS_NUM)
        @ initrv.mean
    )
    _beta = np.array(0.3)
    _t = 0.1
    _m = initrv.mean

    ode_likelihood.check_jacobians(_t, _point, _beta, _m)

    # ##################################################################################
    # BUILD MODEL
    # ##################################################################################

    # ODE measurements
    measurement_matrix_ode = args.ode_measurement_cov * np.eye(STATE_DIM)
    # measurement_matrix_ode_chol_factor = np.sqrt(args.ode_measurement_cov)
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
        measurement_model_ode,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    # Data measurements
    measurement_matrix_data = args.data_measurement_cov * np.eye(OBSERVATION_DIM)
    proj_state_to_IRD = (
        prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0)
        @ prior_transition.proj2process(X_PROCESS_NUM)
    )[1:, :]

    measurement_noiserv_data = randvars.Normal(mean=np.zeros(OBSERVATION_DIM), cov=measurement_matrix_data)
    measurement_model_data = randprocs.markov.discrete.LTIGaussian(
        transition_matrix=proj_state_to_IRD,
        noise=measurement_noiserv_data,
        forward_implementation=forward_implementation,
        backward_implementation=backward_implementation,
    )

    # ##################################################################################
    # Run algorithm
    # ##################################################################################

    data_grid = np.array(train_idcs, copy=True, dtype=np.float64)
    ode_grid = np.arange(*time_domain, step=args.filter_step_size, dtype=np.float64)

    logging.info(f"Solving on time domain: t in {time_domain}")

    merged_locations = probssm.util.unions1d(data_grid, ode_grid)

    merged_observations = []
    merged_measmods = []

    data_idx = 0
    ode_idx = 0

    for loc in merged_locations:
        if np.in1d(loc, data_grid):
            merged_observations.append(SIRD_data[data_idx, 1:])
            merged_measmods.append(measurement_model_data)
            data_idx += 1

        elif np.in1d(loc, ode_grid):
            merged_observations.append(np.array(zero_data))
            merged_measmods.append(linearized_measurement_model_ode)
            ode_idx += 1
        else:
            pass

    logging.info(f"{data_idx} / {merged_locations.size} data locations")
    logging.info(f"{ode_idx} / {merged_locations.size} ODE locations")

    merged_regression_problem = problems.TimeSeriesRegressionProblem(
        observations=merged_observations,
        locations=merged_locations,
        measurement_models=merged_measmods,
    )

    assert len(merged_observations) == len(merged_measmods) == len(merged_locations)

    kalman_filter = filtsmooth.gaussian.Kalman(prior_process)

    logging.info("Computing smoothing posterior ...")
    start_filtsmooth = time.time()
    posterior, _ = kalman_filter.filtsmooth(merged_regression_problem)
    time_filtsmooth = time.time() - start_filtsmooth

    logging.info(
        f"\033[1mFiltering + Smoothing took {time_filtsmooth:.2f} seconds.\033[0m"
    )

    _posterior_save_file = log_dir / "smoothing_posterior_first.npz"

    np.savez(
        _posterior_save_file,
        means=np.stack([s.mean for s in posterior.states]),
        covs=np.stack([s.cov for s in posterior.states]),
    )

    if args.num_samples is not None and args.num_samples > 0:
        logging.info(f"Drawing {args.num_samples} samples from posterior...")

        start_sampling = time.time()
        samples = posterior.sample(rng=rng, size=args.num_samples)
        time_sampling = time.time() - start_sampling

        _samples_save_file = log_dir / "posterior_samples.npy"
        np.save(_samples_save_file, samples)

        logging.info(f"Saved posterior samples to {_samples_save_file}.")

    logging.info(f"\033[1mSampling took {time_sampling:.2f} seconds.\033[0m")

    logging.info("Computation done. Finalize")

    # ##################################################################################
    # Finalize
    # ##################################################################################
    projections_dict = {
        "E_x": prior_transition.proj2process(X_PROCESS_NUM),
        "E_beta": prior_transition.proj2process(BETA_PROCESS_NUM),
        "E0_x": prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0),
        "E0_beta": prior_transition.proj2coord(proc=BETA_PROCESS_NUM, coord=0),
    }
    _projections_save_file = log_dir / "projections.npz"
    np.savez(_projections_save_file, **projections_dict)
    logging.info(f"Saved projections matrices to {_projections_save_file}.")

    data_dict = {
        "sird_data": np.exp(SIRD_data),
        "day_zero": day_zero.to_numpy(),
        "date_range_x": np.array([ts.to_numpy() for ts in date_range_x]),
        "time_domain": np.array(time_domain),
        "data_grid": data_grid,
        "ode_grid": ode_grid,
        "dense_grid": merged_locations,
        "train_idcs": train_idcs,
        "val_idcs": val_idcs,
    }
    _data_info_save_file = log_dir / "data_info.npz"
    np.savez(_data_info_save_file, **data_dict)
    logging.info(f"Saved data info to {_data_info_save_file}.")

    info = {}
    _args_save_file = log_dir / "info.json"
    probssm.args_to_json(_args_save_file, args=args, kwargs=info)
    logging.info(f"Saved info dict to {_args_save_file}.")


if __name__ == "__main__":
    main()
