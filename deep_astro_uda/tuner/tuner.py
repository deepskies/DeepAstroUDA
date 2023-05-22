import numpy as np

try:
    import sklearn
    import sklearn.exceptions
    import sklearn.gaussian_process
except ImportError:  # pragma: no cover
    sklearn = None  # pragma: no cover

try:
    import scipy
    import scipy.optimize
except ImportError:  # pragma: no cover
    scipy = None  # pragma: no cover

from keras_tuner.api_export import keras_tuner_export
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module


# Use 3 times the dimensionality of the space as the default number of
# random points.
dimensions = len(self.hyperparameters.space)
num_initial_points = self.num_initial_points or max(3 * dimensions, 3)
if len(completed_trials) < num_initial_points:
    return self._random_populate_space()

x, y = self._vectorize_trials()

# Ensure no nan, inf in x, y. GPR cannot process nan or inf.
x = np.nan_to_num(x, posinf=0, neginf=0)
y = np.nan_to_num(y, posinf=0, neginf=0)

self.gpr.fit(x, y)

def _upper_confidence_bound(x):
    x = x.reshape(1, -1)
    mu, sigma = self.gpr.predict(x, return_std=True)
    return mu - self.beta * sigma

optimal_val = float("inf")
optimal_x = None
num_restarts = 50
bounds = self._get_hp_bounds()
x_seeds = self._random_state.uniform(
    bounds[:, 0], bounds[:, 1], size=(num_restarts, bounds.shape[0])
)
for x_try in x_seeds:
    # Sign of score is flipped when maximizing.
    result = scipy.optimize.minimize(
        _upper_confidence_bound,
        x0=x_try,
        bounds=bounds,
        method="L-BFGS-B",
    )
    result_fun = (
        result.fun if np.isscalar(result.fun) else result.fun[0]
    )
    if result_fun < optimal_val:
        optimal_val = result_fun
        optimal_x = result.x

values = self._vector_to_values(optimal_x)
return {"status": trial_module.TrialStatus.RUNNING, "values": values}

def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

def get_state(self):
        state = super().get_state()
        state.update(
            {
                "num_initial_points": self.num_initial_points,
                "alpha": self.alpha,
                "beta": self.beta,
            }
        )
        return state

def set_state(self, state):
        super().set_state(state)
        self.num_initial_points = state["num_initial_points"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.gpr = self._make_gpr()

def _vectorize_trials(self):
        x = []
        y = []
        ongoing_trials = set(self.ongoing_trials.values())
        for trial in self.trials.values():
            # Create a vector representation of each Trial's hyperparameters.
            trial_hps = trial.hyperparameters
            vector = []
            for hp in self._nonfixed_space():
                # For hyperparameters not present in the trial (either added
                # after the trial or inactive in the trial), set to default
                # value.
                if (
                    trial_hps.is_active(hp)  # inactive
                    and hp.name in trial_hps.values  # added after the trial
                ):
                    trial_value = trial_hps.values[hp.name]
                else:
                    trial_value = hp.default

                # Embed an HP value into the continuous space [0, 1].
                prob = hp.value_to_prob(trial_value)
                vector.append(prob)

    if trial in ongoing_trials:
        # "Hallucinate" the results of ongoing trials. This ensures that
        # repeat trials are not selected when running distributed.
        x_h = np.array(vector).reshape((1, -1))
        y_h_mean, y_h_std = self.gpr.predict(x_h, return_std=True)
        # Give a pessimistic estimate of the ongoing trial.
        score = y_h_mean[0] + y_h_std[0]
    elif trial.status == "COMPLETED":
        score = trial.score
        # Always frame the optimization as a minimization for
        # scipy.minimize.
        if self.objective.direction == "max":
            score = -1 * score
    elif trial.status in ["FAILED", "INVALID"]:
        # Skip the failed and invalid trials.
        continue

    x.append(vector)
    y.append(score)

x = np.array(x)
y = np.array(y)
return x, y

    def _vector_to_values(self, vector):
        hps = hp_module.HyperParameters()
        vector_index = 0
        for hp in self.hyperparameters.space:
            hps.merge([hp])
            if isinstance(hp, hp_module.Fixed):
                value = hp.value
            else:
                prob = vector[vector_index]
                vector_index += 1
                value = hp.prob_to_value(prob)

            if hps.is_active(hp):
                hps.values[hp.name] = value
        return hps.values

    def _nonfixed_space(self):
        return [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed)
        ]

    def _get_hp_bounds(self):
        bounds = [[0, 1] for _ in self._nonfixed_space()]
        return np.array(bounds)


@keras_tuner_export(
    [
        "keras_tuner.BayesianOptimization",
        "keras_tuner.tuners.BayesianOptimization",
    ]
)
class Optimizer(tuner_module.Tuner):
    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=10,
        num_initial_points=None,
        alpha=1e-4,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs
    ):
        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs
