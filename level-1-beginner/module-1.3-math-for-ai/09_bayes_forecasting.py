"""
09 — Bayesian forecasting with a scalar Kalman filter

The Kalman filter is THE workhorse of Bayesian time-series forecasting.
Strip away the matrices and it's just two Gaussians multiplied together —
i.e. Bayes' rule, applied at every timestep.

We track a noisy 1-D signal (think: drifting temperature sensor):
    state:        x_t = x_{t-1} + drift + w_t,  w_t ~ N(0, Q)   (process noise)
    observation:  z_t = x_t + r_t,              r_t ~ N(0, R)   (measurement noise)

At each step the filter runs PREDICT → UPDATE:

    PREDICT  (extrapolate the state):
        x_pred = x_prev + drift
        P_pred = P_prev + Q

    UPDATE   (apply Bayes' rule with the new measurement z):
        prior      ~ N(x_pred, P_pred)
        likelihood ~ N(z,      R)
        posterior  ~ N(x_post, P_post)   ← also Gaussian (Gaussians are conjugate)

    Combining precisions (1/variance):
        1/P_post = 1/P_pred + 1/R
        x_post   = P_post * (x_pred / P_pred  +  z / R)

    Equivalently, using the Kalman gain K (the standard form):
        K       = P_pred / (P_pred + R)        # how much to trust the measurement
        x_post  = x_pred + K * (z - x_pred)
        P_post  = (1 - K) * P_pred

This is the same predict→update recursion you saw in 08_bayes_rule.py
Exercise 2, just with continuous Gaussian state instead of binary "sick".

Run: python 09_bayes_forecasting.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(seed=1)

# ---------------------------------------------------------------
# 1. Generate a "true" signal and noisy observations
# ---------------------------------------------------------------
T_OBS  = 80                  # number of observed timesteps
T_FORE = 30                  # forecast horizon (no observations here)
DRIFT  = 0.05                # true drift per step
Q_TRUE = 0.10**2             # true process-noise variance
R_TRUE = 0.50**2             # true measurement-noise variance

true_state = np.zeros(T_OBS + T_FORE)
for t in range(1, T_OBS + T_FORE):
    true_state[t] = true_state[t-1] + DRIFT + rng.normal(0, np.sqrt(Q_TRUE))

obs = true_state[:T_OBS] + rng.normal(0, np.sqrt(R_TRUE), size=T_OBS)


# ---------------------------------------------------------------
# 2. Run the scalar Kalman filter on the observations
# ---------------------------------------------------------------
# In real applications you wouldn't know Q or R exactly — you'd estimate
# them. Here we use the true values so the filter is well-tuned.
Q = Q_TRUE
R = R_TRUE
drift = DRIFT

# initial belief: mean 0, very uncertain (large variance)
x_post = 0.0
P_post = 1.0**2

mu_filt    = np.zeros(T_OBS)
sigma_filt = np.zeros(T_OBS)

for t in range(T_OBS):
    # PREDICT — extrapolate using the process model
    x_pred = x_post + drift
    P_pred = P_post + Q

    # UPDATE — combine prediction with the new observation via Bayes
    z = obs[t]
    K = P_pred / (P_pred + R)        # Kalman gain in [0, 1]
    x_post = x_pred + K * (z - x_pred)
    P_post = (1 - K) * P_pred

    mu_filt[t]    = x_post
    sigma_filt[t] = np.sqrt(P_post)

print(f"final filter estimate at t={T_OBS-1}: "
      f"{mu_filt[-1]:.3f} ± {sigma_filt[-1]:.3f}  "
      f"(true: {true_state[T_OBS-1]:.3f})")


# ---------------------------------------------------------------
# 3. Forecast forward (no more observations — only the predict step)
# ---------------------------------------------------------------
# Without measurements to anchor the belief, variance grows linearly
# (one extra Q per step). This is why honest forecasts have widening
# uncertainty bands the further out you go.
mu_fore    = np.zeros(T_FORE)
sigma_fore = np.zeros(T_FORE)

x = mu_filt[-1]
P = sigma_filt[-1]**2
for h in range(T_FORE):
    x = x + drift
    P = P + Q
    mu_fore[h]    = x
    sigma_fore[h] = np.sqrt(P)


# ---------------------------------------------------------------
# 4. Plot truth, observations, filter band, forecast band
# ---------------------------------------------------------------
t_obs  = np.arange(T_OBS)
t_fore = np.arange(T_OBS, T_OBS + T_FORE)
t_true = np.arange(T_OBS + T_FORE)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(t_true, true_state, "k-", linewidth=1.5, label="true state")
ax.scatter(t_obs, obs, s=14, color="gray", alpha=0.6, label="observations")

ax.plot(t_obs, mu_filt, "C0-", linewidth=2, label="filter mean")
ax.fill_between(t_obs,
                mu_filt - 2*sigma_filt, mu_filt + 2*sigma_filt,
                color="C0", alpha=0.2, label="filter ±2σ")

ax.plot(t_fore, mu_fore, "C3-", linewidth=2, label="forecast mean")
ax.fill_between(t_fore,
                mu_fore - 2*sigma_fore, mu_fore + 2*sigma_fore,
                color="C3", alpha=0.2, label="forecast ±2σ")

ax.axvline(T_OBS - 0.5, color="black", linestyle=":", alpha=0.5)
ax.set_xlabel("timestep")
ax.set_ylabel("value")
ax.set_title("Scalar Kalman filter: filtering + forecasting")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
path = out / "09_bayes_forecasting.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {path}")


# ---------------------------------------------------------------
# 5. The Kalman gain in plain English
# ---------------------------------------------------------------
# K = P_pred / (P_pred + R)
#
#   • R small (precise sensor)        → K ≈ 1 → trust the measurement
#   • P_pred small (confident prior)  → K ≈ 0 → ignore the measurement
#
# The filter automatically balances expectation vs. evidence. That's
# Bayesian weighting in closed form — no integrals, no MCMC needed,
# because Gaussians multiplied by Gaussians stay Gaussian.
#
# Generalize to vectors and matrices and you have the full Kalman filter
# used in GPS, drones, control systems, and asset-pricing models.
# Replace "Gaussian" with "particles" and you have a particle filter.
# Make the state discrete and you have a hidden Markov model.
#
# Wherever a system updates its belief over time, this recursion is
# usually hiding inside.
