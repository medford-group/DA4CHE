---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Time Series Models

```{contents}
:local:
:depth: 2
```

:::{admonition} Learning Objectives
:class: tip

By the end of this chapter, you will be able to:
- Remove a linear trend and seasonal component from a time series using polynomial
  fitting and LASSO regression on sine-wave features.
- Construct an autoregressive (AR) model by casting lag features as a standard linear
  regression problem, and use BIC and PACF to select the model order $p$.
- Explain the difference between in-sample (one-step) and dynamic (multi-step) forecasts,
  and understand why dynamic forecasts degrade over time.
- Fit an ARIMA($p,d,q$) model using `statsmodels`, interpret the summary output, and
  generate forecasts with 95% confidence bands.
- Choose between AR, ARIMA, and seasonal ARIMA based on the ACF/PACF patterns and
  properties of the data.
:::

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
import statsmodels.api as sm_api
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use('../settings/plot_style.mplstyle')

clrs = np.array(['#003057', '#EAAA00', '#4B8B9B', '#B3A369', '#377117',
                 '#1879DB', '#8E8B76', '#F5D580', '#002233'])
```

```{code-cell} ipython3
# Reload and prepare the CO₂ dataset (same as Topic 6.1)
df_dow = pd.read_excel('data/impurity_dataset-training.xlsx')
dow_df = df_dow[['Date', 'y:Impurity']].copy()
dow_df['Date'] = pd.to_datetime(dow_df['Date'])
dow_df = dow_df.set_index('Date')

co2_raw = sm_api.datasets.co2.load_pandas().data
co2_df = co2_raw.copy()
co2_df['co2_interp'] = co2_df['co2'].interpolate(method='linear')
co2_df = co2_df[['co2', 'co2_interp']]

y = co2_df['co2_interp']
weeks = np.arange(len(y))
print(f'CO₂ series length: {len(y)} weeks')
```

## Removing Trends

### Linear Trend Fitting

Topic 6.1 showed that the CO₂ series is non-stationary, primarily because of a
rising long-term trend. A natural first step is to fit and subtract that trend.

```{code-cell} ipython3
# Fit a linear trend to the full CO₂ series
m_lin, b_lin = np.polyfit(weeks, y, deg=1)
trend_linear = m_lin * weeks + b_lin
resid_linear = y.values - trend_linear

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(weeks, y.values, label='Data')
axes[0].plot(weeks, trend_linear, '--', label='Linear trend')
axes[0].set_title('CO₂ and linear trend')
axes[0].set_xlabel('Weeks')
axes[0].set_ylabel('CO₂ (ppm)')
axes[0].legend()

axes[1].plot(weeks, resid_linear)
axes[1].set_title('Residuals after linear de-trending')
axes[1].set_xlabel('Weeks')
plt.tight_layout()
```

The residuals after linear de-trending still show a clear oscillating pattern (annual
seasonality) and a slight upward curvature — suggesting the trend is not perfectly
linear. We address the seasonality next.

:::{exercise}
:label: ex-eda-ts-trend-fit

Compare polynomial trend models for the CO₂ series.

1. Fit a degree-1 (linear) and degree-2 (quadratic) polynomial trend to the full
   CO₂ series using `np.polyfit`.
2. Plot both trend lines on top of the data. Which fits the long-run trajectory
   better?
3. Compute the root-mean-square error (RMSE) between the data and each trend.
   Report both values. Does the improvement from degree-1 to degree-2 appear
   physically meaningful?
:::

## Removing Seasonality with LASSO

### Sine-Wave Feature Library

Seasonal patterns are periodic signals. We can model them by generating a library of
sine waves at candidate frequencies and offsets, then using LASSO to select the
combination that best fits the residuals:

$$y_t \approx ax + bx^2 + \sum_{k} c_k \sin\!\left(\frac{\pi x}{\nu_k} - \frac{\phi_k \pi}{\nu_k}\right)$$

where $\nu_k$ are candidate periods (in weeks) and $\phi_k$ are candidate phase offsets.

```{code-cell} ipython3
frequencies = [13, 23, 24, 25, 26, 26.5, 27]
offsets     = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def make_sine_features(x, frequencies, offsets):
    """Generate polynomial and sine-wave features from a 1-D time index."""
    feats  = [x, x**2]
    names  = ['x', 'x^2']
    for freq in frequencies:
        for offset in offsets:
            feats.append(np.sin((np.pi / freq) * x - (offset / freq) * np.pi))
            names.append(f'sin(π·x/{freq} + {offset})')
    return np.column_stack(feats), names

X_all, feat_names = make_sine_features(weeks, frequencies, offsets)
print(f'Feature matrix shape: {X_all.shape}')
```

```{code-cell} ipython3
# Fit LASSO to select the most predictive sine-wave features
lasso_trend = Lasso(alpha=0.1, max_iter=5000)
lasso_trend.fit(X_all, y)
yhat_full = lasso_trend.predict(X_all)
model_resid = y.values - yhat_full

# Report selected features
selected = [(feat_names[i], lasso_trend.coef_[i]) for i in range(len(feat_names))
            if abs(lasso_trend.coef_[i]) > 0]
print(f'Selected {len(selected)} features:')
for name, coef in selected:
    print(f'  {name:45s}  {coef:+.4f}')
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(weeks, y.values, label='Data', alpha=0.6)
axes[0].plot(weeks, yhat_full, '--', label='LASSO fit')
axes[0].set_title('CO₂ and trend + seasonality fit')
axes[0].set_xlabel('Weeks')
axes[0].set_ylabel('CO₂ (ppm)')
axes[0].legend()

axes[1].plot(weeks, model_resid)
axes[1].set_title('Residuals after trend + seasonality removal')
axes[1].set_xlabel('Weeks')
plt.tight_layout()
```

```{code-cell} ipython3
# Verify stationarity of residuals
adf_resid = adfuller(model_resid)
print(f'Residuals ADF p-value: {adf_resid[1]:.4f}  →  P(stationary) ≈ {1 - adf_resid[1]:.4f}')
```

The residuals are now stationary. However, the ACF still shows significant
autocorrelation at many lags — the trend and seasonality have been removed but
short-term temporal dependence remains:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf( model_resid, lags=52, ax=axes[0], title='ACF of residuals')
plot_pacf(model_resid, lags=52, ax=axes[1], title='PACF of residuals')
plt.tight_layout()
```

:::{exercise}
:label: ex-eda-ts-season-lasso

Tune the LASSO regularization for seasonal de-trending.

1. Sweep `alpha` over `[0.01, 0.05, 0.1, 0.5, 1.0]` for `Lasso(..., max_iter=5000)`
   fitted to `X_all` and `y`.
2. For each `alpha`, record the number of selected features (nonzero coefficients)
   and the ADF p-value of the residuals.
3. Plot the number of selected features vs. `alpha` and the ADF p-value vs. `alpha`
   on two subplots. Which `alpha` gives the fewest features while still producing
   stationary residuals (ADF p < 0.05)?
:::

```{code-cell} ipython3
# Apply one round of differencing to further reduce autocorrelation
model_resid_s = pd.Series(model_resid, index=y.index)
resid_diff = (model_resid_s - model_resid_s.shift(1)).dropna()

adf2 = adfuller(resid_diff)
print(f'Differenced residuals ADF p-value: {adf2[1]:.4e}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf( resid_diff, lags=52, ax=axes[0], title='ACF — differenced residuals')
plot_pacf(resid_diff, lags=52, ax=axes[1], title='PACF — differenced residuals')
plt.tight_layout()
```

---

## Auto-Regressive (AR) Models

### From ACF/PACF to Lag Features

An autoregressive model of order $p$ predicts $x_t$ from the $p$ most recent
observations:

$$x_t = b + \sum_{i=1}^{p} w_i x_{t-i} + \varepsilon_t$$

This is simply linear regression where the **features are lagged values of the target**.
We construct the feature matrix by stacking windows of length $p$ from the differenced
residual series:

```{code-cell} ipython3
# Temporal train/test split — all "past" is training, "future" is test
train_ratio = 0.75
N_train = int(train_ratio * len(weeks))
N_test  = len(weeks) - N_train

past_weeks   = weeks[:N_train]
future_weeks = weeks[N_train:]
past_co2     = y.values[:N_train]
future_co2   = y.values[N_train:]

x_diff = resid_diff.values   # differenced residuals, length = len(y) - 1
```

### BIC-Based Order Selection

We sweep $p$ from 1 to 20 and compute the Bayesian Information Criterion (BIC) for
each AR model fitted on the training portion. A lower BIC indicates a better
trade-off between fit quality and model complexity.

```{code-cell} ipython3
def BIC(y_true, y_pred, n_params):
    err = y_true - y_pred
    sigma = np.std(err)
    n = len(y_true)
    return n * np.log(sigma**2 + 1e-12) + n_params * np.log(n)

p_range = range(1, 21)
bic_list = []

for p in p_range:
    AR_X, AR_y = [], []
    for i in range(N_train):
        if i >= p:
            AR_X.append(x_diff[i-p:i])
            AR_y.append(x_diff[i])
    AR_X = np.array(AR_X)
    AR_y = np.array(AR_y)
    arm = LinearRegression().fit(AR_X, AR_y)
    bic_list.append(BIC(AR_y, arm.predict(AR_X), p))

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(list(p_range), bic_list, 'o-')
ax.set_xlabel('AR order p')
ax.set_ylabel('BIC')
ax.set_title('BIC vs. AR order')
best_p = list(p_range)[np.argmin(bic_list)]
ax.axvline(best_p, linestyle='--', color=clrs[1])
ax.set_title(f'BIC vs. AR order (best p = {best_p})')
plt.tight_layout()
print(f'Best AR order by BIC: p = {best_p}')
```

### Fitting the AR Model

```{code-cell} ipython3
p = best_p

AR_X, AR_y = [], []
for i in range(N_train):
    if i >= p:
        AR_X.append(x_diff[i-p:i])
        AR_y.append(x_diff[i])

AR_X = np.array(AR_X)
AR_y = np.array(AR_y)

ARM = LinearRegression().fit(AR_X, AR_y)
print(f'AR({p}) train r²: {ARM.score(AR_X, AR_y):.3f}')
```

The low $r^2$ is expected — after trend and seasonal removal plus differencing, what
remains is largely noise. The AR model captures any residual short-term dependence.

### In-Sample (One-Step) Forecast

For the in-sample check, we feed **actual** past observations into the model at each
step — this is sometimes called a "one-step-ahead" forecast:

```{code-cell} ipython3
# Predict trend + seasonality for the training period
X_past, _ = make_sine_features(past_weeks, frequencies, offsets)
past_trend = lasso_trend.predict(X_past)

# Reconstruct residuals from differenced residuals
past_model_resid = model_resid_s.values[:N_train]

AR_pred_diff = ARM.predict(AR_X)
ar_pred_full = np.zeros(N_train)
ar_pred_full[0] = past_model_resid[0]
for i in range(1, N_train):
    ar_pred_full[i] = ar_pred_full[i-1] + (AR_pred_diff[i-1] if i-1 < len(AR_pred_diff) else 0)

past_predict = past_trend + ar_pred_full

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(past_weeks, past_co2, label='Data', alpha=0.5)
ax.plot(past_weeks, past_trend, '--', label='Trend only')
ax.plot(past_weeks, past_predict, label='AR(p) + trend')
ax.set_xlabel('Weeks')
ax.set_ylabel('CO₂ (ppm)')
ax.set_title('In-sample AR forecast (training period)')
ax.legend()
plt.tight_layout()
```

### Dynamic (Multi-Step) Forecast

For the test period, we no longer have access to the actual future observations, so
each predicted value must feed back as input for the next step — a **dynamic forecast**:

```{code-cell} ipython3
# Seed the dynamic forecast with the last p observed differenced residuals
seed_x = list(x_diff[N_train-p:N_train])

AR_future = []
for i in range(N_test):
    new_X = np.array(seed_x[-p:]).reshape(1, -1)
    next_val = float(ARM.predict(new_X))
    AR_future.append(next_val)
    seed_x.append(next_val)

# Reconstruct future residuals
X_future, _ = make_sine_features(future_weeks, frequencies, offsets)
future_trend = lasso_trend.predict(X_future)

future_resid = np.zeros(N_test)
future_resid[0] = ar_pred_full[-1]
for i in range(1, N_test):
    future_resid[i] = future_resid[i-1] + AR_future[i-1]

future_predict = future_trend + future_resid

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(past_weeks, past_co2, label='Training data', color=clrs[0])
ax.plot(future_weeks, future_co2, label='Test data', color=clrs[2])
ax.plot(past_weeks, past_trend, '--', alpha=0.5, color=clrs[1], label='Trend (train)')
ax.plot(future_weeks, future_trend, '--', alpha=0.5, color=clrs[3], label='Trend (test)')
ax.plot(future_weeks, future_predict, color=clrs[1], label='AR forecast')
ax.set_xlabel('Weeks')
ax.set_ylabel('CO₂ (ppm)')
ax.set_title('Dynamic AR forecast')
ax.legend(fontsize=8)
plt.tight_layout()
```

The dynamic AR forecast quickly reverts to the trend-only prediction because the
differenced residuals look like noise — the AR coefficients cannot sustain a meaningful
multi-step forecast. This is typical: AR models on stationary noise provide little
predictive power beyond a few steps ahead.

:::{note}
**Why does the dynamic forecast fail?** In a dynamic forecast, the model inputs are
its own previous predictions rather than actual observations. Any prediction error
propagates and accumulates at each step. If the true signal is mostly noise (low
autocorrelation after preprocessing), the model quickly converges to predicting the
mean, and the forecast collapses to the trend.
:::

:::{exercise}
:label: ex-eda-ts-ar-order

Explore AR model order selection on the Dow impurity series.

1. Apply one round of differencing to the Dow impurity residuals (after removing
   a linear trend with `np.polyfit`). Verify stationarity with the ADF test.
2. Sweep AR orders $p \in \{1, 2, \ldots, 15\}$ on the training portion
   (first 75%) of the differenced series. Compute and plot the BIC for each order.
3. Fit the best AR model (by BIC) and report the train $r^2$ on the lag-feature matrix.
4. Compute and plot the one-step-ahead in-sample forecast for the training period.
   Does the AR model capture any residual structure beyond the trend?
:::

---

## ARIMA Models

### The ARIMA Framework

Manually de-trending, removing seasonality, differencing, and fitting an AR model is
instructive, but tedious. The **ARIMA($p, d, q$)** framework packages these steps
into a single model:

| Parameter | Meaning | How to choose |
|---|---|---|
| $p$ | AR order — lags in the autoregressive term | PACF of differenced series |
| $d$ | Integration order — number of differences needed for stationarity | ADF test (typically $d=1$ for most economic/physical series) |
| $q$ | MA order — lags in the moving-average (residual) term | ACF of differenced series |

The **moving-average (MA)** component uses past *forecast errors* rather than past
observations:

$$x_t = \mu + \varepsilon_t + \sum_{i=1}^{q} \theta_i \varepsilon_{t-i}$$

Combining AR($p$), integration ($d$), and MA($q$) gives ARIMA($p,d,q$), implemented
in `statsmodels.tsa.arima.model.ARIMA`.

### Choosing $(p, d, q)$

We already know $d = 1$ (one difference makes CO₂ stationary). To find $p$ and $q$,
inspect the ACF and PACF of the once-differenced series:

```{code-cell} ipython3
diffed = (co2_df['co2_interp'] - co2_df['co2_interp'].shift(1)).dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf( diffed, lags=52, ax=axes[0], title='ACF — first difference')
plot_pacf(diffed, lags=52, ax=axes[1], title='PACF — first difference')
plt.tight_layout()
```

- **PACF** cuts off sharply after lag 3–5 → $p \approx 4$
- **ACF** cuts off sharply after lag 3–4 → $q \approx 4$

### Fitting ARIMA

```{code-cell} ipython3
from statsmodels.tsa.arima.model import ARIMA

# Temporal split
train_co2 = co2_df['co2_interp'].iloc[:N_train]
test_co2  = co2_df['co2_interp'].iloc[N_train:N_train + N_test]

arima = ARIMA(train_co2, order=(4, 1, 4))
arima_fit = arima.fit()
print(arima_fit.summary())
```

```{code-cell} ipython3
# In-sample fit
fig, ax = plt.subplots(figsize=(10, 4))
train_co2.plot(ax=ax, label='Training data', alpha=0.6)
arima_fit.fittedvalues.plot(ax=ax, label='ARIMA in-sample fit')
ax.set_title('ARIMA(4,1,4) — in-sample fit')
ax.legend()
plt.tight_layout()

mae_insample = np.mean(np.abs(arima_fit.resid))
print(f'In-sample mean absolute error: {mae_insample:.3f} ppm')
```

### Dynamic Forecast with Uncertainty Bands

```{code-cell} ipython3
forecast_result = arima_fit.get_forecast(steps=N_test)
fc_mean = forecast_result.predicted_mean
fc_ci   = forecast_result.conf_int(alpha=0.05)   # 95% CI

fig, ax = plt.subplots(figsize=(10, 4))
train_co2.plot(ax=ax, label='Training data', color=clrs[0])
test_co2.plot(ax=ax, label='Test data', color=clrs[2])
fc_mean.plot(ax=ax, label='ARIMA forecast', color=clrs[1])
ax.fill_between(fc_ci.index,
                fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                alpha=0.2, color=clrs[1], label='95% CI')
ax.set_title('ARIMA(4,1,4) — dynamic forecast')
ax.legend(fontsize=8)
plt.tight_layout()
```

The forecast captures the general upward trend at first, but the oscillating seasonal
pattern fades because the standard ARIMA model does not explicitly model seasonality.
The 95% confidence bands widen appropriately over the forecast horizon — correctly
reflecting the growing uncertainty of longer-range predictions.

:::{note}
**Seasonal ARIMA (SARIMA)** extends the model to explicitly handle periodic
seasonality, adding seasonal AR and MA terms at a fixed period $s$ (e.g., $s=52$
for annual patterns in weekly data). The model is written SARIMA($p,d,q$)($P,D,Q$)$_s$.
For the CO₂ dataset, a SARIMA model with $s=52$ would preserve the annual oscillation
throughout the forecast horizon. This is beyond the scope of this course, but the
statsmodels `SARIMAX` class implements it directly.
:::

:::{exercise}
:label: ex-eda-ts-arima-dow

Apply an ARIMA model to the Dow impurity series.

1. Use the Augmented Dickey-Fuller test to determine $d$ (the number of differences
   needed for stationarity).
2. Inspect the ACF and PACF of the differenced series to estimate $p$ and $q$.
3. Fit `ARIMA(p, d, q)` on the first 75% of the Dow data (chronological split).
4. Generate a dynamic forecast for the remaining 25%. Plot the forecast with 95%
   confidence bands overlaid on the actual test data.
5. Report the mean absolute error (MAE) on the test set. How does it compare to
   just using the training mean as the forecast?
:::

---

## Practical Guidance

The table below summarizes the main modeling choices for univariate time series:

| Situation | Recommended approach |
|---|---|
| Clear linear trend, strong seasonal pattern, abundant data | SARIMA or Prophet (beyond scope) |
| Clear trend, moderate autocorrelation after differencing | ARIMA |
| Stationary after 1 difference, PACF cuts off sharply | AR($p$) with manual lag features |
| No obvious trend, low autocorrelation | Standard regression ignoring time order may suffice |
| Non-stationary, complex multi-frequency seasonality | Remove trend/seasonality with LASSO features, then AR or ARIMA on residuals |

**Common pitfalls:**
- Using $r^2$ on a non-stationary series is misleading — a model that just predicts
  the trend achieves high $r^2$ without capturing any dynamics.
- Forgetting to respect chronological order in train/test splits leads to data leakage.
- Over-differencing (applying more differences than necessary) destroys useful signal.
- Dynamic forecasts always degrade — report confidence intervals rather than treating
  long-range forecasts as reliable point estimates.

:::{exercise}
:label: ex-eda-ts-model-select

Use the practical guidance table to choose and evaluate a model for a new scenario.

Given the Dow impurity series:

1. Check whether the original (undifferenced) series is stationary using the ADF
   test. Based on the result and the ACF/PACF patterns, select a row from the
   guidance table above and state which model you would recommend.
2. Implement your recommended model and generate a dynamic forecast for the last
   25% of the series (chronological split). Plot the forecast with 95% confidence
   bands if applicable.
3. Compute the mean absolute error (MAE) of the forecast and compare it to the
   naive baseline (predicting the training-set mean for all future values).
:::

---

## Summary

- Removing **trend and seasonality** (via polynomial + LASSO sine-wave fitting) isolates
  the residual stochastic component that is the target for AR/ARIMA modeling.

- **AR($p$) models** cast time series prediction as standard linear regression on lag
  features. The order $p$ is selected using the PACF or BIC sweep.

- **In-sample (one-step) forecasts** use actual past observations as inputs and look
  excellent; **dynamic (multi-step) forecasts** feed back predictions and degrade
  quickly as errors accumulate — this gap is a key diagnostic of how much genuine
  predictability exists beyond the trend.

- **ARIMA($p,d,q$)** packages differencing ($d$), autoregression ($p$), and
  moving-average error terms ($q$) into a single framework. The parameters are chosen
  from PACF (→ $p$), ADF test (→ $d$), and ACF (→ $q$). Statsmodels provides forecast
  confidence intervals, which are essential for communicating uncertainty.

- For seasonal data like CO₂, **SARIMA** extends ARIMA with seasonal terms; for
  purely trend-plus-noise industrial data like Dow impurity, standard ARIMA or AR
  models are usually sufficient.

## Additional Reading

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.),
  Chapters 8–9 — ARIMA and SARIMA in depth with R examples; most concepts transfer
  directly to Python/statsmodels: [otexts.com/fpp3](https://otexts.com/fpp3/)
- statsmodels documentation:
  [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html),
  [SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- Seabold, S. & Perktold, J. (2010), "statsmodels: Econometric and statistical
  modeling with Python," *Proceedings of the 9th Python in Science Conference*
