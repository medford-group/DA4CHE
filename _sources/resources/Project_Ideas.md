```{contents}
:local:
:depth: 2
```

# Project Ideas

These are directions, not specifications. Each one names a question worth asking, a dataset
from {doc}`Datasets </resources/Datasets>` that can support it, and the chapters that bear
on it. None of them is fully specified, and that is intentional — deciding what exactly to
predict, and what counts as success, is the part that teaches you the most.

A direction becomes a project when you can state three things in one sentence each: the
question, the quantity you will report, and the comparison that tells you whether the answer
is any good.

## Directions

| Direction | Question | Dataset | Chapters |
|---|---|---|---|
| Power plant output regression | How accurately can net electrical output be predicted from four ambient measurements, and where does the model stop working? | Combined Cycle Power Plant | 1.3, 2.2, 2.3 |
| Process fault detection | Can faults be detected and identified from correlated sensor readings, and how quickly? | Tennessee Eastman | 5.2, 3.1–3.3, 6.1 |
| Battery cycle life prediction | Can long-run cycle life be predicted from the first few cycles alone? | Li-ion battery cycle life | 2.4, 2.5 |
| Fed-batch optimization | Which control trajectory maximizes yield, and how confident can you be given batch-to-batch variability? | IndPenSim penicillin | 1.4, 4.4, 6.1 |
| Imbalanced quality classification | Can rare manufacturing failures be flagged without drowning operators in false alarms? | SECOM | 3.1, 3.2, 3.4 |
| Remaining useful life | How far ahead of failure can degradation be detected, and does the model survive a shift in operating conditions? | NASA C-MAPSS | 6.1, 6.2, 2.2 |
| Refinery unit modeling | Can a data-driven model of an FCC unit be accurate enough to support a control decision? | FCC operating data | 2.6, 6.2 |
| Structure-to-property prediction | How much does the choice of molecular representation matter compared to the choice of model? | QM9 | 2.4, 6.3, 6.4 |
| Polymer property prediction | Can polymerization outcomes be predicted well enough to narrow an experimental search? | Polymerization property data | 2.6, 6.3 |
| Property prediction with uncertainty | Can a thermophysical property be predicted with an honest uncertainty estimate and a stated valid range? | NIST ThermoML | 1.5, 2.2 |
| Inverse design | Given a target output, can a generative model propose inputs that produce it? | Light-color inverse problem | 5.4, 1.4 |
| Single-particle tracking in video | Can nanoparticles be segmented and tracked through a noisy video well enough that the recovered trajectories support a diffusion measurement? | SAM-EM liquid-phase TEM videos | 6.4, 6.1, 2.2 |
| Benchmark model comparison | Do the ranking differences between models on a benchmark survive a change in the evaluation split? | Matbench | 2.2, 2.3 |

## Three worked examples

The tables above are deliberately terse. These three show what filling one in looks like.
They are chosen to span the book: one straightforward regression, one classification problem
where the validation strategy is the hard part, and one where there is very little data.

### 1. Predicting combined cycle power plant output

**Question.** How accurately can net hourly electrical energy output be predicted from
ambient temperature, ambient pressure, relative humidity, and exhaust vacuum — and over
what range of conditions is that prediction trustworthy?

**Data.** The Combined Cycle Power Plant dataset: 9,568 hourly averages, four inputs, one
output. Small, complete, and real.

**A path through it.** Start with the scatter plots. Ambient temperature and exhaust vacuum
are strongly correlated with output and with each other, which is worth understanding before
you fit anything. Fit an ordinary least squares model as the baseline and record its error —
this is the number every later model has to beat. Then add nonlinear terms
({doc}`Nonlinear Feature Engineering </2-regression/Topic2.6-Nonlinear_Feature_Engineering>`)
or move to a non-parametric model
({doc}`Non-parametric Models </2-regression/Topic2.1-Non-parametric_Models>`), using
cross-validation to choose complexity rather than choosing it by eye
({doc}`Complexity Optimization </2-regression/Topic2.3-Complexity_Optimization>`).

**What to report.** A parity plot and a residual plot for the baseline and for your best
model. One error measure, chosen and justified before you look at the results. A statement
of the input range the model was trained on.

**Where it gets hard.** The interesting failure is extrapolation. Hold out the hottest and
coldest ambient conditions rather than a random subset, retrain, and see what happens. A
model that looks excellent under random splitting can degrade sharply at the edges of the
operating envelope, and that is exactly the regime where a plant engineer would want to use
it.

### 2. Fault detection on the Tennessee Eastman process

**Question.** Can process faults be detected from 52 correlated sensor and manipulated
variables, can the fault *type* be identified, and how many samples after fault onset does
detection take?

**Data.** The Tennessee Eastman simulation: 500 runs, 52 variables, 20 fault types.

**A path through it.** Reduce first. With 52 correlated variables, PCA
({doc}`Dimensionality Reduction </5-exploratory_data_analysis/Topic5.2-Dimensionality_Reduction>`)
both compresses the input and gives you the classical monitoring statistics as a baseline
detector. Then treat identification as a classification problem
({doc}`Classification Basics </3-classification/Topic3.1-Classification_Basics>` through
{doc}`Alternate Classification Models </3-classification/Topic3.3-Alternate_Classification_Models>`),
starting with a linear classifier before anything more elaborate.

**What to report.** False alarm rate, missed detection rate, and detection delay — three
numbers, not one accuracy. A confusion matrix over fault types, which will show that some
faults are nearly indistinguishable from each other and from normal operation.

**Where it gets hard.** The validation strategy, not the model. Consecutive samples within a
run are highly autocorrelated, so a random split puts near-copies of training points into
the test set and produces accuracies in the high nineties that mean nothing. Split by run,
and within a run respect time order
({doc}`Time Series Basics </6-advanced_topics/Topic6.1-Time_Series_Basics>`). The gap between
the random-split number and the split-by-run number is the single most instructive result
this dataset will give you.

### 3. Battery cycle life from early cycles

**Question.** Using measurements from only the first few charge–discharge cycles, can the
eventual cycle life of a cell be predicted?

**Data.** The Li-ion battery cycle life dataset — commercial cells cycled to failure. Note
the shape: on the order of a hundred cells, and as many candidate features as you care to
compute from the early-cycle voltage and capacity curves.

**A path through it.** This is a high-dimensional, small-sample problem, which is a
different game from the first example. Feature construction comes first: summarize each
cell's early cycles into a handful of physically motivated numbers. Then use regularization
and dimensionality reduction to avoid fitting noise
({doc}`High-dimensional Data </2-regression/Topic2.4-High_dimensional_data>` and
{doc}`High-dimensional Regression </2-regression/Topic2.5-High_dimensional_regression>`).

**What to report.** Cross-validated error with an honest accounting of how many models you
tried, and which features the final model relies on. With this few samples, the feature
ranking is itself a result worth reporting — and worth checking for stability across folds.

**Where it gets hard.** With roughly a hundred samples and dozens of features, almost any
flexible model can fit the training data perfectly, and small changes to the split can
reorder your feature importances entirely. Refit on several different splits before you
believe any claim about which features matter. This is the clearest demonstration in the
whole catalog of why more capacity is not more knowledge.

## Where each stage is covered

| Stage | Chapters |
|---|---|
| Retrieving data from APIs and the web | {doc}`4.2 Online Data Access </4-data_management/Topic4.2-Online_Data_Access>`, {doc}`4.3 Accessing Data with AI Tools </4-data_management/Topic4.3-Accessing_Data_with_AI_Tools>` |
| Cleaning, indexing, missing values, storage | {doc}`4.1 Data Organization </4-data_management/Topic4.1-Data_Organization>`, {doc}`4.4 Complex Structured Data </4-data_management/Topic4.4-Complex_Structured_Data>` |
| Exploring and visualizing before modeling | {doc}`5.1 High Dimensional Data </5-exploratory_data_analysis/Topic5.1-High_Dimensional_Data>`, {doc}`5.2 Dimensionality Reduction </5-exploratory_data_analysis/Topic5.2-Dimensionality_Reduction>`, {doc}`5.3 Clustering </5-exploratory_data_analysis/Topic5.3-Clustering>` |
| Baseline regression and honest validation | {doc}`1.3 Linear Regression </1-numerical_methods/Topic1.3-Linear_Regression>`, {doc}`2.2 Model Validation </2-regression/Topic2.2-Model_Validation>` |
| Fitting physical models and estimating parameters | {doc}`1.4 Numerical Optimization </1-numerical_methods/Topic1.4-Numerical_Optimization>`, {doc}`1.5 Parameter Estimation </1-numerical_methods/Topic1.5-Parameter_Estimation>` |
| Nonlinear and high-dimensional regression | {doc}`2.4 High-dimensional Data </2-regression/Topic2.4-High_dimensional_data>`, {doc}`2.5 High-dimensional Regression </2-regression/Topic2.5-High_dimensional_regression>`, {doc}`2.6 Nonlinear Feature Engineering </2-regression/Topic2.6-Nonlinear_Feature_Engineering>` |
| Classification, including imbalanced problems | {doc}`3.1 Classification Basics </3-classification/Topic3.1-Classification_Basics>` through {doc}`3.4 High-dimensional Classification </3-classification/Topic3.4-High-dimensional_Classification>` |
| Time series and forecasting | {doc}`6.1 Time Series Basics </6-advanced_topics/Topic6.1-Time_Series_Basics>`, {doc}`6.2 Time Series Models </6-advanced_topics/Topic6.2-Time_Series_Models>` |
| Generative models and inverse problems | {doc}`5.4 Generative Models </5-exploratory_data_analysis/Topic5.4-Generative_Models>` |
| Neural networks | {doc}`6.3 Neural Network Basics </6-advanced_topics/Topic6.3-Neural_Network_Basics>`, {doc}`6.4 Neural Network Architectures </6-advanced_topics/Topic6.4-Neural_Network_Architectures>` |

## What a finished piece of work looks like

Independently of the direction you take, the same handful of things distinguish analysis
that can be trusted from analysis that merely produced a number:

- **A notebook that runs top to bottom.** From data loading to final figure, on a fresh
  kernel, without manual intervention. If it only works when cells are run out of order,
  the result is not reproducible even by you.
- **A baseline you can point at.** Whatever your final model is, the reader needs to know
  what a simple alternative achieved.
- **Parity and residual plots.** A single error metric hides structure; residuals against
  each input reveal it. Systematic curvature in a residual plot is a finding, not a nuisance.
- **An error measure chosen on purpose.** Say why root mean squared error rather than mean
  absolute error, or why recall matters more than accuracy for your problem, before you
  report it.
- **A validation strategy that matches the intended use.** If the model would be deployed on
  a future batch, validate on a future batch.
- **A stated boundary.** Where should this model not be used — outside what input range, on
  what equipment, under what conditions? A model without a stated limit is a model whose
  limits have not been examined.
