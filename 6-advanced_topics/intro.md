# Advanced Topics

This module introduces two areas where specialized techniques significantly outperform
the general-purpose methods covered in earlier modules: **time series analysis** and
**neural networks**.

**Time series data** (process measurements, environmental monitoring, sensor streams)
violate the independence assumption that underlies most regression and classification
methods. Topics 6.1 and 6.2 cover the statistical properties of time series, methods
for handling non-stationarity and seasonality, and classical forecasting models (AR
and ARIMA).

**Neural networks** learn hierarchical feature representations directly from data,
overcoming the limitations of hand-crafted feature engineering for complex,
high-dimensional inputs. Topic 6.3 builds intuition from first principles — a single
neuron, activation functions, backpropagation — and demonstrates the `MLPRegressor`
on familiar datasets. Topic 6.4 surveys three widely-used architectures (CNN, LSTM,
autoencoder) through minimal PyTorch implementations, connecting each to earlier
material: CNNs to LDA/PCA projections, LSTMs to ARIMA forecasting, and autoencoders
to the generative models of Module 5.

The overarching theme is that **architecture is structured feature engineering**:
every design choice encodes an assumption about what structure in the data is worth
exploiting. Making those choices thoughtfully — and knowing when simpler classical
methods suffice — is a core data analytics skill.
