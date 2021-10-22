import numpy as np
import probssm


class LogisticLikelihood(object):
    def __init__(self, prior, ode_parameters) -> None:
        self.prior = prior

        self.a = ode_parameters["a"]
        self.b = ode_parameters["b"]

        self.rhs = probssm.ivp.log_rhs
        self.drhs_dx = probssm.ivp.log_jac

    def check_jacobians(self, t, point, a, b, m):
        """Check jacobians by finite differences."""

        x_fct = lambda eps: self.rhs(t, y=point + eps, a=a, b=b)
        meas_fct = lambda eps: self.measure_ode(t, m + eps)

        x_jac = self.drhs_dx(t, y=point, a=a, b=b)
        meas_jac = self.measure_ode_jacobian(t, m)

        probssm.test_jacobian(dim=point.size, jacobian=x_jac, function=x_fct)
        probssm.test_jacobian(dim=m.size, jacobian=meas_jac, function=meas_fct)

    def measure_ode(self, t, state):
        x = state

        E0_x = self.prior.proj2coord(coord=0)
        E1_x = self.prior.proj2coord(coord=1)

        return (E1_x @ x) - self.rhs(t, y=E0_x @ x, a=self.a, b=self.b)

    def measure_ode_jacobian(self, t, state):
        x = state

        E0_x = self.prior.proj2coord(coord=0)
        E1_x = self.prior.proj2coord(coord=1)

        dx_dx = E1_x - self.drhs_dx(t, y=E0_x @ x, a=self.a, b=self.b) @ E0_x

        return dx_dx


class LVLikelihood(object):
    def __init__(
        self,
        prior,
        alpha_link_fn,
        alpha_link_fn_deriv,
        beta_link_fn,
        beta_link_fn_deriv,
        gamma_link_fn,
        gamma_link_fn_deriv,
        delta_link_fn,
        delta_link_fn_deriv,
    ) -> None:
        self.prior = prior

        self.alpha_link_fn = alpha_link_fn
        self.alpha_link_fn_deriv = alpha_link_fn_deriv
        self.beta_link_fn = beta_link_fn
        self.beta_link_fn_deriv = beta_link_fn_deriv
        self.gamma_link_fn = gamma_link_fn
        self.gamma_link_fn_deriv = gamma_link_fn_deriv
        self.delta_link_fn = delta_link_fn
        self.delta_link_fn_deriv = delta_link_fn_deriv

        self.rhs = probssm.ivp.lv_rhs
        self.drhs_duv = probssm.ivp.lv_jac_x
        self.drhs_dalpha = probssm.ivp.lv_jac_alpha
        self.drhs_dbeta = probssm.ivp.lv_jac_beta
        self.drhs_dgamma = probssm.ivp.lv_jac_gamma
        self.drhs_ddelta = probssm.ivp.lv_jac_delta

    def check_jacobians(self, t, point, alpha, beta, gamma, delta, m):
        """Check jacobians by finite differences."""

        x_fct = lambda eps: self.rhs(
            t,
            y=point + eps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        alpha_fct = lambda eps: self.rhs(
            t,
            y=point,
            alpha=alpha + eps,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        beta_fct = lambda eps: self.rhs(
            t,
            y=point,
            alpha=alpha,
            beta=beta + eps,
            gamma=gamma,
            delta=delta,
        )
        gamma_fct = lambda eps: self.rhs(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma + eps,
            delta=delta,
        )
        delta_fct = lambda eps: self.rhs(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta + eps,
        )
        meas_fct = lambda eps: self.measure_ode(t, m + eps)

        x_jac = self.drhs_duv(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        alpha_jac = self.drhs_dalpha(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        beta_jac = self.drhs_dbeta(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        gamma_jac = self.drhs_dgamma(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        delta_jac = self.drhs_ddelta(
            t,
            y=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        meas_jac = self.measure_ode_jacobian(t, m)

        probssm.test_jacobian(dim=point.size, jacobian=x_jac, function=x_fct)
        probssm.test_jacobian(dim=1, jacobian=alpha_jac, function=alpha_fct)
        probssm.test_jacobian(dim=1, jacobian=beta_jac, function=beta_fct)
        probssm.test_jacobian(dim=1, jacobian=gamma_jac, function=gamma_fct)
        probssm.test_jacobian(dim=1, jacobian=delta_jac, function=delta_fct)
        probssm.test_jacobian(dim=m.size, jacobian=meas_jac, function=meas_fct)

    def measure_ode(self, t, state):
        x = self.prior.proj2process(0) @ state
        alpha = self.prior.proj2process(1) @ state
        beta = self.prior.proj2process(2) @ state
        gamma = self.prior.proj2process(3) @ state
        delta = self.prior.proj2process(4) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_alpha = self.prior.proj2coord(proc=1, coord=0)
        E0_beta = self.prior.proj2coord(proc=2, coord=0)
        E0_gamma = self.prior.proj2coord(proc=3, coord=0)
        E0_delta = self.prior.proj2coord(proc=4, coord=0)

        return (E1_x @ x) - self.rhs(
            t,
            y=(E0_x @ x),
            alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
            beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
            gamma=self.gamma_link_fn(E0_gamma @ gamma).squeeze(),
            delta=self.delta_link_fn(E0_delta @ delta).squeeze(),
        )

    def measure_ode_jacobian(self, t, state):
        x = self.prior.proj2process(0) @ state
        alpha = self.prior.proj2process(1) @ state
        beta = self.prior.proj2process(2) @ state
        gamma = self.prior.proj2process(3) @ state
        delta = self.prior.proj2process(4) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_alpha = self.prior.proj2coord(proc=1, coord=0)
        E0_beta = self.prior.proj2coord(proc=2, coord=0)
        E0_gamma = self.prior.proj2coord(proc=3, coord=0)
        E0_delta = self.prior.proj2coord(proc=4, coord=0)

        dx_dx = (
            E1_x
            - self.drhs_duv(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma_link_fn(E0_gamma @ gamma).squeeze(),
                delta=self.delta_link_fn(E0_delta @ delta).squeeze(),
            )
            @ E0_x
        )

        dx_dalpha = (
            -self.drhs_dalpha(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma_link_fn(E0_gamma @ gamma).squeeze(),
                delta=self.delta_link_fn(E0_delta @ delta).squeeze(),
            )
            * self.alpha_link_fn_deriv((E0_alpha @ alpha).squeeze()).reshape(-1, 1)
            @ E0_alpha
        )

        dx_dbeta = (
            -self.drhs_dbeta(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma_link_fn(E0_gamma @ gamma).squeeze(),
                delta=self.delta_link_fn(E0_delta @ delta).squeeze(),
            )
            * self.beta_link_fn_deriv((E0_beta @ beta).squeeze()).reshape(-1, 1)
            @ E0_beta
        )

        dx_dgamma = (
            -self.drhs_dgamma(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma_link_fn(E0_gamma @ gamma).squeeze(),
                delta=self.delta_link_fn(E0_delta @ delta).squeeze(),
            )
            * self.gamma_link_fn_deriv((E0_gamma @ gamma).squeeze()).reshape(-1, 1)
            @ E0_gamma
        )

        dx_ddelta = (
            -self.drhs_ddelta(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma_link_fn(E0_gamma @ gamma).squeeze(),
                delta=self.delta_link_fn(E0_delta @ delta).squeeze(),
            )
            * self.delta_link_fn_deriv((E0_delta @ delta).squeeze()).reshape(-1, 1)
            @ E0_delta
        )

        return np.hstack([dx_dx, dx_dalpha, dx_dbeta, dx_dgamma, dx_ddelta])


class VDPLikelihood(object):
    def __init__(
        self,
        prior,
        alpha_link_fn,
        alpha_link_fn_deriv,
    ) -> None:
        self.prior = prior

        self.alpha_link_fn = alpha_link_fn
        self.alpha_link_fn_deriv = alpha_link_fn_deriv

        self.rhs = probssm.ivp.vanderpol_rhs
        self.drhs_dx = probssm.ivp.vanderpol_jac_x
        self.drhs_dalpha = probssm.ivp.vanderpol_jac_alpha

    def check_jacobians(self, t, point, alpha, m):
        """Check jacobians by finite differences."""

        x_fct = lambda eps: self.rhs(
            t,
            y=point + eps,
            alpha=alpha,
        )
        alpha_fct = lambda eps: self.rhs(
            t,
            y=point,
            alpha=alpha + eps,
        )
        meas_fct = lambda eps: self.measure_ode(t, m + eps)

        x_jac = self.drhs_dx(
            t,
            y=point,
            alpha=alpha,
        )
        alpha_jac = self.drhs_dalpha(
            t,
            y=point,
            alpha=alpha,
        )
        meas_jac = self.measure_ode_jacobian(t, m)

        probssm.test_jacobian(dim=point.size, jacobian=x_jac, function=x_fct)
        probssm.test_jacobian(dim=1, jacobian=alpha_jac, function=alpha_fct)
        probssm.test_jacobian(dim=m.size, jacobian=meas_jac, function=meas_fct)

    def measure_ode(self, t, state):
        x = self.prior.proj2process(0) @ state
        alpha = self.prior.proj2process(1) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_alpha = self.prior.proj2coord(proc=1, coord=0)

        return (E1_x @ x) - self.rhs(
            t,
            y=(E0_x @ x),
            alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
        )

    def measure_ode_jacobian(self, t, state):
        x = self.prior.proj2process(0) @ state
        alpha = self.prior.proj2process(1) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_alpha = self.prior.proj2coord(proc=1, coord=0)

        dx_dx = (
            E1_x
            - self.drhs_dx(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
            )
            @ E0_x
        )

        dx_dalpha = (
            -self.drhs_dalpha(
                t,
                y=(E0_x @ x),
                alpha=self.alpha_link_fn(E0_alpha @ alpha).squeeze(),
            )
            * self.alpha_link_fn_deriv((E0_alpha @ alpha).squeeze()).reshape(-1, 1)
            @ E0_alpha
        )

        return np.hstack([dx_dx, dx_dalpha])


class SIRDLikelihood(object):
    def __init__(self, prior, ode_parameters, beta_link_fn, beta_link_fn_deriv) -> None:

        self.prior = prior

        self.gamma = ode_parameters["gamma"]
        self.eta = ode_parameters["eta"]
        self.population_count = ode_parameters["population_count"]

        self.beta_link_fn = beta_link_fn
        self.beta_link_fn_deriv = beta_link_fn_deriv

        self.rhs = probssm.ivp.sird_rhs
        self.drhs_dsird = probssm.ivp.sird_jac_x
        self.drhs_dbeta = probssm.ivp.sird_jac_beta

    def check_jacobians(self, t, point, beta, m):
        """Check jacobians by finite differences."""

        x_fct = lambda eps: self.rhs(
            t,
            y=point + eps,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        beta_fct = lambda eps: self.rhs(
            t,
            y=point,
            beta=beta + eps,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

        meas_fct = lambda eps: self.measure_ode(t, m + eps)

        x_jac = self.drhs_dsird(
            t,
            y=point,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        beta_jac = self.drhs_dbeta(
            t,
            y=point,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

        meas_jac = self.measure_ode_jacobian(t, m)

        probssm.test_jacobian(dim=point.size, jacobian=x_jac, function=x_fct)
        probssm.test_jacobian(dim=1, jacobian=beta_jac, function=beta_fct)
        probssm.test_jacobian(dim=m.size, jacobian=meas_jac, function=meas_fct)

    def measure_ode(self, t, state):
        x = self.prior.proj2process(0) @ state
        beta = self.prior.proj2process(1) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_beta = self.prior.proj2coord(proc=1, coord=0)

        return (E1_x @ x) - self.rhs(
            t,
            E0_x @ x,
            beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

    def measure_ode_jacobian(self, t, state):

        x = self.prior.proj2process(0) @ state
        beta = self.prior.proj2process(1) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_beta = self.prior.proj2coord(proc=1, coord=0)

        dx_dx = (
            E1_x
            - self.drhs_dsird(
                t,
                E0_x @ x,
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            @ E0_x
        )
        dx_dbeta = (
            -self.drhs_dbeta(
                t,
                E0_x @ x,
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            * self.beta_link_fn_deriv(E0_beta @ beta).squeeze()
            @ E0_beta
        )

        return np.hstack([dx_dx, dx_dbeta])


class LogSIRDLikelihood(object):
    def __init__(self, prior, ode_parameters, beta_link_fn, beta_link_fn_deriv) -> None:

        self.prior = prior

        self.gamma = ode_parameters["gamma"]
        self.eta = ode_parameters["eta"]
        self.population_count = ode_parameters["population_count"]

        self.beta_link_fn = beta_link_fn
        self.beta_link_fn_deriv = beta_link_fn_deriv

        self.rhs = probssm.ivp.sird_rhs
        self.drhs_dsird = probssm.ivp.sird_jac_x
        self.drhs_dbeta = probssm.ivp.sird_jac_beta

    def check_jacobians(self, t, point, beta, m):
        """Check jacobians by finite differences."""

        x_fct = lambda eps: self.rhs(
            t,
            y=point + eps,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        beta_fct = lambda eps: self.rhs(
            t,
            y=point,
            beta=beta + eps,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

        meas_fct = lambda eps: self.measure_ode(t, m + eps)

        x_jac = self.drhs_dsird(
            t,
            y=point,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        beta_jac = self.drhs_dbeta(
            t,
            y=point,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

        meas_jac = self.measure_ode_jacobian(t, m)

        probssm.test_jacobian(dim=point.size, jacobian=x_jac, function=x_fct)
        probssm.test_jacobian(dim=1, jacobian=beta_jac, function=beta_fct)
        probssm.test_jacobian(dim=m.size, jacobian=meas_jac, function=meas_fct)

    def measure_ode(self, t, state):
        x = self.prior.proj2process(0) @ state
        beta = self.prior.proj2process(1) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_beta = self.prior.proj2coord(proc=1, coord=0)

        return (np.exp(E0_x @ x) * (E1_x @ x)) - self.rhs(
            t,
            y=np.exp(E0_x @ x),
            beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

    def measure_ode_jacobian(self, t, state):

        x = self.prior.proj2process(0) @ state
        beta = self.prior.proj2process(1) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_beta = self.prior.proj2coord(proc=1, coord=0)

        dx_dx = (
            ((np.exp(E0_x @ x).reshape(-1, 1) * E0_x) * (E1_x @ x).reshape(-1, 1))
            + (np.exp(E0_x @ x).reshape(-1, 1) * E1_x)
            - self.drhs_dsird(
                t,
                y=np.exp(E0_x @ x),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            @ (np.exp(E0_x @ x).reshape(-1, 1) * E0_x)
        )
        dx_dbeta = (
            -self.drhs_dbeta(
                t,
                y=np.exp(E0_x @ x),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            * self.beta_link_fn_deriv(E0_beta @ beta).squeeze()
            @ E0_beta
        )

        return np.hstack([dx_dx, dx_dbeta])


class SIRDVLikelihood(object):
    def __init__(self, prior, ode_parameters, beta_link_fn, beta_link_fn_deriv) -> None:

        self.prior = prior

        self.gamma = ode_parameters["gamma"]
        self.eta = ode_parameters["eta"]
        self.population_count = ode_parameters["population_count"]

        self.beta_link_fn = beta_link_fn
        self.beta_link_fn_deriv = beta_link_fn_deriv

        self.rhs = probssm.ivp.sirdv_rhs
        self.drhs_dsird = probssm.ivp.sirdv_jac_x
        self.drhs_dbeta = probssm.ivp.sirdv_jac_beta
        self.drhs_dv = probssm.ivp.sirdv_jac_vacc

    def check_jacobians(self, t, point, vacc, beta, m):
        """Check jacobians by finite differences."""

        x_fct = lambda eps: self.rhs(
            t,
            y=point + eps,
            V=vacc,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        beta_fct = lambda eps: self.rhs(
            t,
            y=point,
            V=vacc,
            beta=beta + eps,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        vacc_fct = lambda eps: self.rhs(
            t,
            y=point,
            V=vacc + eps,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        meas_fct = lambda eps: self.measure_ode(t, m + eps)

        x_jac = self.drhs_dsird(
            t,
            y=point,
            V=vacc,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        beta_jac = self.drhs_dbeta(
            t,
            y=point,
            V=vacc,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        vacc_jac = self.drhs_dv(
            t,
            y=point,
            V=vacc,
            beta=beta,
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )
        meas_jac = self.measure_ode_jacobian(t, m)

        probssm.test_jacobian(dim=point.size, jacobian=x_jac, function=x_fct)
        probssm.test_jacobian(dim=1, jacobian=beta_jac, function=beta_fct)
        probssm.test_jacobian(dim=1, jacobian=vacc_jac, function=vacc_fct)
        probssm.test_jacobian(dim=m.size, jacobian=meas_jac, function=meas_fct)

    def measure_ode(self, t, state):
        x = self.prior.proj2process(0) @ state
        beta = self.prior.proj2process(1) @ state
        vacc = self.prior.proj2process(2) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_beta = self.prior.proj2coord(proc=1, coord=0)
        E0_vacc = self.prior.proj2coord(proc=2, coord=0)

        return (np.exp(E0_x @ x) * (E1_x @ x)) - self.rhs(
            t,
            y=np.exp(E0_x @ x),
            V=np.exp(E0_vacc @ vacc).squeeze(),
            beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
            gamma=self.gamma,
            eta=self.eta,
            population_count=self.population_count,
        )

    def measure_ode_jacobian(self, t, state):

        x = self.prior.proj2process(0) @ state
        beta = self.prior.proj2process(1) @ state
        vacc = self.prior.proj2process(2) @ state

        E0_x = self.prior.proj2coord(proc=0, coord=0)
        E1_x = self.prior.proj2coord(proc=0, coord=1)
        E0_beta = self.prior.proj2coord(proc=1, coord=0)
        E0_vacc = self.prior.proj2coord(proc=2, coord=0)

        dx_dx = (
            ((np.exp(E0_x @ x).reshape(-1, 1) * E0_x) * (E1_x @ x).reshape(-1, 1))
            + (np.exp(E0_x @ x).reshape(-1, 1) * E1_x)
            - self.drhs_dsird(
                t,
                y=np.exp(E0_x @ x),
                V=np.exp(E0_vacc @ vacc).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            @ (np.exp(E0_x @ x).reshape(-1, 1) * E0_x)
        )
        dx_dbeta = (
            -self.drhs_dbeta(
                t,
                y=np.exp(E0_x @ x),
                V=np.exp(E0_vacc @ vacc).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            * self.beta_link_fn_deriv(E0_beta @ beta).squeeze()
            @ E0_beta
        )

        dx_dvacc = (
            -self.drhs_dv(
                t,
                y=np.exp(E0_x @ x),
                V=np.exp(E0_vacc @ vacc).squeeze(),
                beta=self.beta_link_fn(E0_beta @ beta).squeeze(),
                gamma=self.gamma,
                eta=self.eta,
                population_count=self.population_count,
            )
            @ (np.exp(E0_vacc @ vacc).reshape(-1, 1) * E0_vacc)
        )

        return np.hstack([dx_dx, dx_dbeta, dx_dvacc])
