"""SDE models as transitions."""

import functools

import numpy as np
import probnum as pn
import scipy.linalg


class StackedTransition(pn.randprocs.markov.continuous.LTISDE):
    def __init__(
        self, transitions, forward_implementation="sqrt", backward_implementation="sqrt"
    ):
        self.transitions = tuple(transitions)
        self.dimensions = tuple((t.state_dimension for t in self.transitions))
        self.total_dimension = sum(self.dimensions)

        pn.randprocs.markov.continuous.LTISDE.__init__(
            self,
            drift_matrix=self._drift_matrix,
            force_vector=self._force_vector,
            dispersion_matrix=self._dispersion_matrix,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

    @functools.cached_property
    def _drift_matrix(self):
        return scipy.linalg.block_diag(*(t.drift_matrix for t in self.transitions))

    @functools.cached_property
    def _force_vector(self):
        return np.zeros(self.total_dimension)

    @functools.cached_property
    def _dispersion_matrix(self):
        return scipy.linalg.block_diag(*(t.dispersion_matrix for t in self.transitions))

    def proj2process(self, num):
        start = sum(self.dimensions[0:num]) if num > 0 else 0
        stop = start + self.dimensions[num] if num < len(self.transitions) else None
        return np.eye(self.total_dimension)[start:stop, :]

    def proj2coord(self, proc, coord):
        if isinstance(proc, int):
            process = self.transitions[proc]
        else:
            raise TypeError(f"Invalid type {type(proc)} provided.")

        return process.proj2coord(coord)

    @functools.cached_property
    def state_idcs(self):
        idx_dicts = []
        for num, p in enumerate(self.transitions):
            idx_dict = {}
            for q in range(p.num_derivatives + 1):
                projmat = self.proj2coord(num, coord=q)
                flattened_projmat = projmat.sum(0)
                idx_offset = sum(self.dimensions[0:num]) if num > 0 else 0
                idx_dict[f"state_d{q}"] = np.nonzero(flattened_projmat)[0] + idx_offset
            idx_dicts.append(idx_dict)

        return idx_dicts

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Discretise and propagate
        discretised_model = self.discretise(dt=dt)
        rv, info = discretised_model.forward_rv(
            rv, t, compute_gain=compute_gain, _diffusion=_diffusion
        )

        return rv, info

    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        **kwargs,
    ):
        if dt is None:
            raise ValueError(
                "Continuous-time transitions require a time-increment ``dt``."
            )

        # Discretise and propagate
        discretised_model = self.discretise(dt=dt)
        rv, info = discretised_model.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            rv_forwarded=rv_forwarded,
            gain=gain,
            t=t,
            _diffusion=_diffusion,
        )

        return rv, info
