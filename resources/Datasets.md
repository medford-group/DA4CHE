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

A few entries are published in repositories that carry **no license file**. They are
included because the data is genuinely useful and its provenance is clear, but the absence
is called out in the table, and it means you should ask the authors before redistributing
the data or releasing anything derived from it.

The **Data** column records how each dataset was produced: *Measured* for instrument,
plant, or laboratory data; *Simulated* for the output of a model, whether a process
simulator, a quantum-chemistry calculation, or a building-energy model; *Mixed* where a
single collection holds both; and *See source* where the source does not say plainly
enough to label it. The distinction matters more than it looks. Simulated data has no
sensor dropout, no drift, and no unexplained outliers, so a method that works on it has
not yet met the thing that makes real data hard.

The scale, license, and provenance details below are summarized from each source and have
**not** all been independently verified — see
{doc}`how this section was compiled </resources/intro>` before relying on them.
:::

## Process data and time series

These are the datasets that look most like industrial practice: many correlated sensors,
autocorrelation in time, batch or run structure, missing values, and faults. They are also
the ones where a careless random train/test split will quietly invalidate your results.

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **Combined Cycle Power Plant** | Measured | Multivariate regression; a clean first target with real measurements | 9,568 rows; 4 inputs (ambient temperature, ambient pressure, relative humidity, exhaust vacuum), 1 output (net hourly electrical energy output) | [UCI 294](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) · [10.1016/j.ijepes.2014.02.027](https://doi.org/10.1016/j.ijepes.2014.02.027) |
| **Tennessee Eastman process simulation** | Simulated | Fault detection and classification, PCA, anomaly detection, time-aware validation | 500 runs, 52 variables, 20 fault types. Simulated, so it is clean by the standards of real plant data; split by run *and* by time | [10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1) |
| **PRONTO multiphase flow benchmark** | Measured | Process monitoring and fault detection with heterogeneous data types | About 1.7 GB across several data types — meaningfully more setup than a CSV | [Zenodo 1341583](https://zenodo.org/records/1341583) |
| **Fluid catalytic cracking operating data** | See source | Data-driven modeling and control of a refinery unit | Released with the paper; see its data availability statement | [10.1016/j.compchemeng.2022.107900](https://doi.org/10.1016/j.compchemeng.2022.107900) |
| **IndPenSim penicillin fed-batch** | Simulated | Bioprocess modeling, batch-wise validation, control and optimization studies | Industrial-scale simulated fed-batch fermentation, released as multiple batches | [Mendeley Data](https://data.mendeley.com/datasets/pdnjz7zz5x/1) · simulator paper [10.1016/j.jbiotec.2014.10.029](https://doi.org/10.1016/j.jbiotec.2014.10.029) |
| **SECOM semiconductor manufacturing** | Measured | Imbalanced classification, missing data, aggressive feature selection | 1,567 rows × 591 sensor features. Many missing values and very few failure cases — the class imbalance *is* the problem. CC BY 4.0 | [UCI 179](https://archive.ics.uci.edu/dataset/179/secom) |
| **NASA C-MAPSS turbofan degradation** | Simulated | Remaining useful life, predictive maintenance, distribution shift between train and test | 708 training and 707 test engine trajectories across four sub-datasets (FD001–FD004) with different fault modes and operating conditions. The catalog does not state an explicit license | [data.nasa.gov](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) · [Saxena et al. 2008](https://ntrs.nasa.gov/citations/20190001645) |
| **Paracetamol batch crystallization under control** | Measured | Batch-structured time series; soft-sensor regression; data from a closed control loop | Five experimental controlled runs, 243–772 rows × 19 columns each, plus separate instrument training sets (1,054 rows). Every run is a batch, so a split that mixes runs leaks. Small enough to open directly in pandas. MIT | [GitHub](https://github.com/fearrais96/Batch-Crystallization-Experimental-Control) · [10.1021/acs.iecr.5c03894](https://doi.org/10.1021/acs.iecr.5c03894) |
| **Nuclear waste slurry fault detection** | Measured | Fault detection on measured rather than simulated data; fusing three instruments | Raman, ATR-FTIR, and particle-size measurements from simulant slurry processing runs, distributed as one 7 MB archive alongside the analysis scripts. A measured counterpart to Tennessee Eastman, and a rare thing: real process data with real faults in it. MIT | [GitHub](https://github.com/magrover/MSPM-Fault-Detection) · [10.1002/aic.70234](https://doi.org/10.1002/aic.70234) |

## Spectroscopy and soft sensors

Spectra are the most common process analytical measurement in chemical engineering, and they
have a shape almost nothing else in this catalog has: far more features than samples. A few
dozen spectra against sixteen hundred Raman shifts is ordinary. You cannot fit ordinary
least squares to data like that at all, which makes these the natural companions to
{doc}`High-dimensional Data </2-regression/Topic2.4-High_dimensional_data>`,
{doc}`High-dimensional Regression </2-regression/Topic2.5-High_dimensional_regression>`, and
{doc}`Dimensionality Reduction </5-exploratory_data_analysis/Topic5.2-Dimensionality_Reduction>`.

The recurring application is the **soft sensor**: an instrument measures something fast and
easy — a spectrum, a chord length distribution — and a calibration model converts it into the
quantity you actually care about, such as a concentration or a crystal size distribution. The
model *is* the sensor, which means how you validate it is a process decision and not merely a
statistic.

Most entries below come from the Grover lab at Georgia Tech, whose data repositories are
collected at [github.com/magrover](https://github.com/magrover).

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **Paracetamol ATR-FTIR in mixed solvent** | Measured | Calibration model development; PLS against PCA + neural network; in-run calibration strategy | 247 spectra × 464 wavenumber channels, each labeled with ethanol weight fraction, temperature, and paracetamol concentration. Spans ethanol/water mixtures and deliberately includes probe-to-probe variability, which is what makes the calibration hard. CC BY 4.0 | [GitHub](https://github.com/fearrais96/IR-Calibration-Model-Framework) · [10.1021/acs.oprd.5c00338](https://doi.org/10.1021/acs.oprd.5c00338) |
| **Nuclear waste simulant FTIR** | Measured | Quantification when unmodeled species are present; blind source separation, MCR-ALS, PLS | 3,902 spectra × 155 wavenumbers from a 2 L non-radioactive process run, plus 8 training spectra, 5 single-species references, and ion-chromatography concentrations for nitrate and nitrite. The README documents every file and column — rarer than it should be. **No license file** | [GitHub](https://github.com/magrover/Blind_Source_Separation_CLS.PCA.MCR-ALS) · [10.3389/fnuen.2023.1295995](https://doi.org/10.3389/fnuen.2023.1295995) |
| **Multicomponent slurry quantification** | Measured | Comparing two instruments on one problem; multi-species quantification in a dense slurry | 48 ATR-FTIR spectra × 243 wavenumbers and 66 Raman spectra × 1,601 shifts, each paired with gravimetric concentrations for five species. The two instruments have different sample sets, so they cannot simply be concatenated. **No license file** | [GitHub](https://github.com/magrover/multicomponent-slurry-quantification) · [10.1021/acs.iecr.3c01249](https://doi.org/10.1021/acs.iecr.3c01249) |

## Images and microscopy

The only image data in the chapters of this book is the 8 × 8 handwritten digit set built into
scikit-learn, which stands in for image data rather than being an example of it. The datasets
below are the real thing, and they come in two flavors: still images of crystals captured
through an in-line process probe, and video of nanoparticles moving in a liquid cell. Both
connect most directly to
{doc}`Neural Network Architectures </6-advanced_topics/Topic6.4-Neural_Network_Architectures>`.

The four still-image sets are annotated for a specific task, and they carry two access
conditions worth settling before you plan around them: downloading requires a free Kaggle
account, and all four are CC BY-SA 4.0, whose ShareAlike clause constrains what you may do
with a derived dataset. All four come from **OpenCrystalData**, a maintained open-access
database with Georgia Tech co-authors. Start with the EasyViewer set — it is an order of
magnitude smaller than the largest.

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **EasyViewer in-line images** | Measured | A first look at image-based process monitoring, at a size you can actually download | 120 in-situ images of wollastonite and L-glutamic acid, with size and chord length information. 0.45 GB — the smallest of the four and the place to start. CC BY-SA 4.0 | [Kaggle](https://www.kaggle.com/datasets/opencrystaldata/easyviewer-based-image-characterization) |
| **Crystallization impurity detection** | Measured | Anomaly and morphology detection — a classification problem where the classes are shapes | 400 raw and 6,000 cropped in-situ images of a cephalexin monohydrate slurry, captured with an EasyViewer-100 probe, with phenylglycine impurity crystals present in some. 1.47 GB. CC BY-SA 4.0 | [Kaggle](https://www.kaggle.com/datasets/opencrystaldata/cephalexin-reactive-crystallization) |
| **Standard polystyrene microspheres** | Measured | Object detection and segmentation against particle-level ground truth | 2,300 in-situ images of standard spheres with particle-level annotations. Known, uniform geometry is what makes this the fair test case before you attempt real crystals. 1.72 GB. CC BY-SA 4.0 | [Kaggle](https://www.kaggle.com/datasets/opencrystaldata/standard-polystyrene-microspheres-polys) |
| **Ag-Crystal needle images** | Measured | Estimating a particle size distribution from images and checking it against offline measurement | 3,888 in-situ images of needle-like crystals from an industrial agrochemical process, intended for models whose PSD output is comparable with offline PSD. Needles are far harder than spheres. 5.31 GB — the largest of the four. CC BY-SA 4.0 | [Kaggle](https://www.kaggle.com/datasets/opencrystaldata/agcrystal-images) |

All four are described together in
[10.1016/j.dche.2024.100150](https://doi.org/10.1016/j.dche.2024.100150), which is the paper
to cite and the best guide to what each one contains. Note that the per-dataset DOIs printed
in that paper did not resolve when this section was last checked — use the Kaggle links above.

The video set below is a different problem. A still image asks *what is in this frame*; a
video asks that plus *which object here is the same object as in the last one*, and that
second question is where the interesting failures live.

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **SAM-EM liquid-phase TEM videos** | Mixed | Instance segmentation and single-particle tracking, against ground truth that carries identity across frames | 1,000 simulated liquid-cell TEM videos: 50,000 grayscale frames (1024 × 1024, 0.25 nm per pixel) with 50,000 matching instance masks, spanning liquid layer thickness 5–170 nm and electron flux 12.5–187.5. Plus a 3,600-frame held-out set over 12 layer and flux combinations, and two real experimental videos (237 and 75 frames) that carry no masks. 102 GB in total, but every piece downloads separately and the useful ones are small. No license file | [10.57967/hf/9700](https://doi.org/10.57967/hf/9700) |

**Download one piece, not the repository.** That 102 GB is the whole release, most of it a
64 GB ablation training set and a 2.7 GB model checkpoint. Individual files can be downloaded
on their own, and the two to start with are `holdoutValidationSet-4ParticlesOverlapping`
(30 MB: one 50-frame video of four overlapping particles, with masks — note that it is a zip
archive despite having no file extension) and `experimentalAggregation.zip` (3 MB: 237 frames
of a real aggregation event, without masks). The full labeled validation set,
`holdoutValidationSet.zip`, is 2.3 GB.

**The annotations are instance masks, not binary masks.** Each is a PNG the size of its frame:
black background, one distinct color per particle, and the color assigned to a particle holds
across every frame of its video. Two consequences worth planning around. You can recover
per-particle trajectories from the labels alone, with no model involved, which makes this a
mean-squared-displacement problem as much as a segmentation one
({doc}`Time Series Basics </6-advanced_topics/Topic6.1-Time_Series_Basics>`). And a model can
be scored on whether it kept each particle's identity, not just on how well its outlines
overlap — which is the harder question, and the one this dataset exists to ask.

**The labeled videos are simulated and the real videos are unlabeled.** Every mask here
belongs to a simulated video; the two experimental videos ship as raw frames. So a supervised
result is trained and validated entirely on simulator output, and the experimental frames are
where you find out whether it transferred. The caution in the note at the top of this page
applies with unusual force, because here the simulator is what defines the noise you are
learning to see through.

The paper is [10.1039/d6dd00211k](https://doi.org/10.1039/d6dd00211k) (*Digital Discovery*,
2026, from a Georgia Tech group). The analysis code and a desktop application are at
[JamaliLab/SAM-EM](https://github.com/JamaliLab/SAM-EM) under Apache 2.0, and that repository
carries a 50-frame worked example, so you can see the file layout before downloading anything.
The data itself has **no license file and no dataset card** — its documentation lives in the
paper and the code repository rather than alongside the data — so ask the authors before
redistributing it or releasing anything derived from it.

## Materials and molecules

Structure-to-property problems, where the interesting question is usually how you represent
the input rather than which regressor you use. These connect most directly to
{doc}`High-dimensional Data </2-regression/Topic2.4-High_dimensional_data>` and
{doc}`Neural Network Architectures </6-advanced_topics/Topic6.4-Neural_Network_Architectures>`.

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **QM9** | Simulated | Structure-to-property mapping; comparing molecular representations | About 134,000 small organic molecules with properties computed by DFT — atomization energy, internal energy, enthalpy, band gap, dipole moment | [10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22) · [quantum-machine.org](https://quantum-machine.org/datasets/) |
| **Matbench** | Mixed | Materials regression and classification, and *fair* model comparison | 13 tasks spanning roughly 300 to 132,000 samples, most of them computed; three are measured — experimental band gap, experimental metallicity, and steel yield strength. Check the terms of each underlying source dataset | [Matbench](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench) |
| **Open Reaction Database** | Measured | Reaction outcome prediction, experiment planning, working with a real schema | More than 2 million reactions. Data CC BY-SA 4.0, software Apache 2.0. Preparing it is more work than reading a CSV | [open-reaction-database.org](https://open-reaction-database.org/about) |
| **Polymerization property data** | See source | Predictive models for polymer properties and polymerization processes | See the paper's data availability statement | [10.1039/D4PY00995A](https://doi.org/10.1039/D4PY00995A) |
| **Li-ion battery cycle life** | Measured | Predicting long-run behavior from early-cycle features; high-dimensional regression at small sample size | Commercial cells cycled to failure, with early-cycle features used to predict eventual cycle life. Few samples, many candidate features | [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8) · related [battery-parameter-spaces](https://github.com/petermattia/battery-parameter-spaces) |
| **PET stabilizer additives** | Mixed | Small-data classification and feature selection where the features vastly outnumber the samples | 59 additives × 1,875 computed molecular descriptors, labeled with six measured stabilizer performance columns (dry at 1–3 hours, wet at 3–15 days). The extreme shape — thirty times more descriptors than samples — *is* the problem, and a companion file adds MACCS fingerprints for 10,000 candidate molecules. **No license file** | [GitHub](https://github.com/magrover/PET_additive_classification) · [10.1021/acsapm.0c00921](https://doi.org/10.1021/acsapm.0c00921) |

## Thermophysical properties

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **NIST ThermoML Archive** | Measured | Property prediction, and practice with structured XML, units, uncertainty, and source tracking | A large XML archive. Availability and reuse terms depend on the original publisher | [ThermoML](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive) |
| **UCI Energy Efficiency** | Simulated | Regression, model comparison, residual diagnostics | 768 rows, 8 inputs, 2 targets, produced by building-energy simulation of 768 candidate building shapes. Small and simple by design — a good warm-up, not a process dataset. CC BY 4.0 | [UCI 242](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| **Henry's law constants** | Mixed | Regression against temperature across a very wide range of solutes | A large curated compilation drawn from the literature — predominantly measured values, alongside estimated and calculated ones | [10.5194/acp-23-10901-2023](https://doi.org/10.5194/acp-23-10901-2023) |

## Inverse problems and generative modeling

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **Light-color inverse problem** | Measured | Demonstrating and *evaluating* generative models on an inverse problem with a known forward map | Measured light intensity as a function of red, green, and blue input settings. Small, physically interpretable, and the forward direction is easy — which is what makes the inverse direction a fair test | [10.1039/D5DD00137D](https://doi.org/10.1039/D5DD00137D) |

## Benchmarks for evaluating models

| Dataset | Data | Good for | Scale and notes | Source |
|---|---|---|---|---|
| **ChemBench** | — | Testing what a language model actually knows about chemistry; comparing tool use against no tool use | More than 2,700 questions. Intended for evaluation, **not** for training | [ChemBench](https://lamalab-org.github.io/chembench/) |

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
