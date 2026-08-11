```{contents}
:local:
:depth: 2
```

# Datasets

Every dataset below is openly available, has a citable source, and has enough
documentation that you can tell what the columns mean. They are grouped by flavor rather
than by method, because the useful question is "what kind of data is this?" before it is
"what model should I fit?"

Scale and license are listed wherever the source states them. Check both before you plan
around a dataset — a benchmark licensed for evaluation only, or a 1.7 GB multi-format
download, changes what the work looks like.

:::{note}
None of these are used in the chapters of this book, and none are derived from it. That is
deliberate: working on data you have not already seen analyzed is most of the point.

The scale and license details below are summarized from each source and have **not** all
been independently verified — see
{doc}`how this section was compiled </resources/intro>` before relying on them.
:::

## Process data and time series

These are the datasets that look most like industrial practice: many correlated sensors,
autocorrelation in time, batch or run structure, missing values, and faults. They are also
the ones where a careless random train/test split will quietly invalidate your results.

| Dataset | Good for | Scale and notes | Source |
|---|---|---|---|
| **Combined Cycle Power Plant** | Multivariate regression; a clean first target with real measurements | 9,568 rows; 4 inputs (ambient temperature, ambient pressure, relative humidity, exhaust vacuum), 1 output (net hourly electrical energy output) | [UCI 294](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) · [10.1016/j.ijepes.2014.02.027](https://doi.org/10.1016/j.ijepes.2014.02.027) |
| **Tennessee Eastman process simulation** | Fault detection and classification, PCA, anomaly detection, time-aware validation | 500 runs, 52 variables, 20 fault types. Simulated, so it is clean by the standards of real plant data; split by run *and* by time | [10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1) |
| **PRONTO multiphase flow benchmark** | Process monitoring and fault detection with heterogeneous data types | About 1.7 GB across several data types — meaningfully more setup than a CSV | [Zenodo 1341583](https://zenodo.org/records/1341583) |
| **Fluid catalytic cracking operating data** | Data-driven modeling and control of a refinery unit | Released with the paper; see its data availability statement | [10.1016/j.compchemeng.2022.107900](https://doi.org/10.1016/j.compchemeng.2022.107900) |
| **IndPenSim penicillin fed-batch** | Bioprocess modeling, batch-wise validation, control and optimization studies | Industrial-scale simulated fed-batch fermentation, released as multiple batches | [Mendeley Data](https://data.mendeley.com/datasets/pdnjz7zz5x/1) · simulator paper [10.1016/j.jbiotec.2014.10.029](https://doi.org/10.1016/j.jbiotec.2014.10.029) |
| **SECOM semiconductor manufacturing** | Imbalanced classification, missing data, aggressive feature selection | 1,567 rows × 591 sensor features. Many missing values and very few failure cases — the class imbalance *is* the problem. CC BY 4.0 | [UCI 179](https://archive.ics.uci.edu/dataset/179/secom) |
| **NASA C-MAPSS turbofan degradation** | Remaining useful life, predictive maintenance, distribution shift between train and test | 708 training and 707 test engine trajectories across four sub-datasets (FD001–FD004) with different fault modes and operating conditions. The catalog does not state an explicit license | [data.nasa.gov](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) · [Saxena et al. 2008](https://ntrs.nasa.gov/citations/20190001645) |

## Materials and molecules

Structure-to-property problems, where the interesting question is usually how you represent
the input rather than which regressor you use. These connect most directly to
{doc}`High-dimensional Data </2-regression/Topic2.4-High_dimensional_data>` and
{doc}`Neural Network Architectures </6-advanced_topics/Topic6.4-Neural_Network_Architectures>`.

| Dataset | Good for | Scale and notes | Source |
|---|---|---|---|
| **QM9** | Structure-to-property mapping; comparing molecular representations | About 134,000 small organic molecules with computed properties — atomization energy, internal energy, enthalpy, band gap, dipole moment | [10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22) · [quantum-machine.org](https://quantum-machine.org/datasets/) |
| **Matbench** | Materials regression and classification, and *fair* model comparison | 13 tasks spanning roughly 300 to 132,000 samples. Check the terms of each underlying source dataset | [Matbench](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench) |
| **Open Reaction Database** | Reaction outcome prediction, experiment planning, working with a real schema | More than 2 million reactions. Data CC BY-SA 4.0, software Apache 2.0. Preparing it is more work than reading a CSV | [open-reaction-database.org](https://open-reaction-database.org/about) |
| **Polymerization property data** | Predictive models for polymer properties and polymerization processes | See the paper's data availability statement | [10.1039/D4PY00995A](https://doi.org/10.1039/D4PY00995A) |
| **Li-ion battery cycle life** | Predicting long-run behavior from early-cycle features; high-dimensional regression at small sample size | Commercial cells cycled to failure, with early-cycle features used to predict eventual cycle life. Few samples, many candidate features | [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8) · related [battery-parameter-spaces](https://github.com/petermattia/battery-parameter-spaces) |

## Thermophysical properties

| Dataset | Good for | Scale and notes | Source |
|---|---|---|---|
| **NIST ThermoML Archive** | Property prediction, and practice with structured XML, units, uncertainty, and source tracking | A large XML archive. Availability and reuse terms depend on the original publisher | [ThermoML](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive) |
| **UCI Energy Efficiency** | Regression, model comparison, residual diagnostics | 768 rows, 8 inputs, 2 targets. Small and simple by design — a good warm-up, not a process dataset. CC BY 4.0 | [UCI 242](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| **Henry's law constants** | Regression against temperature across a very wide range of solutes | A large curated compilation drawn from the literature | [10.5194/acp-23-10901-2023](https://doi.org/10.5194/acp-23-10901-2023) |

## Inverse problems and generative modeling

| Dataset | Good for | Scale and notes | Source |
|---|---|---|---|
| **Light-color inverse problem** | Demonstrating and *evaluating* generative models on an inverse problem with a known forward map | Measured light intensity as a function of red, green, and blue input settings. Small, physically interpretable, and the forward direction is easy — which is what makes the inverse direction a fair test | [10.1039/D5DD00137D](https://doi.org/10.1039/D5DD00137D) |

## Benchmarks for evaluating models

| Dataset | Good for | Scale and notes | Source |
|---|---|---|---|
| **ChemBench** | Testing what a language model actually knows about chemistry; comparing tool use against no tool use | More than 2,700 questions. Intended for evaluation, **not** for training | [ChemBench](https://lamalab-org.github.io/chembench/) |

## Finding your own data

The catalog above is a starting point, not a boundary. Two habits make independent
searching far more productive.

**Read the data availability statement first.** Most journals now require one, and it is
the fastest way to tell whether a paper's data is genuinely obtainable. It will point you
at a repository record, a supporting information file, or — often enough — a sentence
saying the data are available on request, which in practice means budget for a delay. Check
this before you get attached to the idea.

**Prefer repositories that give you a stable identifier.** A DOI or a numbered record means
the data you download today is the data someone else downloads next year.

General-purpose and chemistry-specific places to look:

| Source | What you will find |
|---|---|
| [Awesome Industrial Datasets](https://github.com/jonathanwvd/awesome-industrial-datasets) ([browsable version](https://www.indatlas.com/datasets/)) | The best single index for this domain: around 190 industrial datasets, each tagged with domain, modality, task, access route, size, year, and license. Roughly twenty are chemical and process datasets, and several of the entries above are catalogued there with more metadata than their original hosts provide |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets) | Several hundred curated, well-documented datasets with stated licenses; several of the entries above live here |
| [Zenodo](https://zenodo.org/) | General-purpose research data with DOIs, including a great deal of supplementary data from papers |
| [Harvard Dataverse](https://dataverse.harvard.edu/) | Institutional repository with DOIs; hosts the Tennessee Eastman record above |
| [Materials Project](https://next-gen.materialsproject.org/) | Computed properties for a very large number of inorganic materials, with a documented API |
| [NIST Standard Reference Data](https://www.nist.gov/srd) and [WebBook](https://webbook.nist.gov/chemistry/) | Authoritative thermophysical and spectroscopic data, with uncertainties |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | Chemical structures, properties, and bioassay data, with a REST API |

Several of these are exercised directly in
{doc}`Online Data Access </4-data_management/Topic4.2-Online_Data_Access>` and
{doc}`Accessing Data with AI Tools </4-data_management/Topic4.3-Accessing_Data_with_AI_Tools>`,
so the mechanics of pulling from an API and parsing the response are already covered.
