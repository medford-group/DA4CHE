# Module 4: Data Management

Real-world data science is rarely limited by the choice of model — it is almost
always limited by data quality. Before any regression or classification algorithm
can be applied, the raw data must be collected, organized, cleaned, and stored in
a form that supports efficient analysis.

This module covers the practical skills needed to manage data in chemical engineering
contexts. We use the **Dow Chemical distillation column dataset** — a real industrial
time-series with sensor readings, missing values, and outliers — as a running example
throughout.

## Topics

- **Topic 4.1 — Data Organization**: Pandas indexing and filtering, handling missing
  values, outlier detection, and efficient storage with HDF5.

- **Topic 4.2 — Online Data Access**: Retrieving data from web APIs, parsing JSON
  responses, and working with databases programmatically.

## Dataset

The **Dow impurity dataset** (`impurity_dataset-training.xlsx`) contains time-stamped
sensor measurements from a primary distillation column, including reflux flow rates,
feed flows, and bed temperatures. The target variable `y:Impurity` is the product
impurity level. This dataset was provided by Dow Chemical for educational use.
