# Datasets and Resources

The chapters in this book teach methods using a small number of datasets chosen because
they are convenient to explain: the Dow distillation impurity data, an ethanol IR
spectrum, a perovskite stability table, a handful of synthetic examples. That is the right
choice for teaching a method, but it is a poor model of what independent work feels like.
Real analysis starts with a question and a dataset that was not curated for you, and most
of the difficulty lives in the gap between them.

This section is a bridge across that gap. It is not an assignment and carries no
requirements — it is a catalog of data worth working with, a set of directions worth
taking, and pointers to the maintained resource collections that go well beyond anything
listed here.

:::{admonition} How this section was compiled
:class: warning

The pages in this section were assembled with substantial help from AI tools —
specifically Claude Code, which collected the entries from a set of papers and DOIs I had
gathered, from the CACHE resource lists linked in
{doc}`External Resources </resources/Resources>`, and from each dataset's own
documentation, and then summarized them into the tables you see here.

Links were checked automatically at the time of writing, and the descriptions reflect what
each source says about itself. But **I have not independently verified every dataset, every
license, or every claim made about them here.** Sizes, licenses, and characterizations of
what a dataset contains should be treated as a starting point rather than as authoritative.
Before you rely on a dataset — and especially before you use one in published work — check
its license, citation requirements, and documentation at the source.

If you find an error, a dead link, a mislabeled license, or a dataset that is not what it is
described as, please [open an issue](https://github.com/medford-group/DA4CHE/issues).
Corrections are genuinely welcome, and are the fastest way for this section to improve.
:::

## What is here

- **{doc}`Datasets </resources/Datasets>`** — a curated catalog of chemical-engineering
  datasets that are open, documented, and sized for independent work, with the license and
  scale information you need in order to decide whether a dataset fits before you commit
  to it. It closes with a short guide to finding your own.

- **{doc}`Project Ideas </resources/Project_Ideas>`** — a table of concrete directions,
  each paired with a dataset and the chapters that bear on it, plus three worked examples
  showing what a well-scoped piece of work actually looks like end to end.

- **{doc}`External Resources </resources/Resources>`** — the CACHE Teaching Resources
  Center collections, other courses and books, material on generative AI and agents, and
  a short checklist for using any of it responsibly.

## Choosing a dataset

Most of the ways a project goes wrong are decided before any modeling happens, when the
dataset is chosen. Four questions are worth answering first:

- **Can you actually get it?** A dataset behind a registration wall, a request form, or a
  1.7 GB download you have nowhere to put is a dataset you will not use. Download it
  before you plan around it.
- **Can you get it again?** Prefer data you can reload from a stable source — a DOI, a
  numbered repository record, an API you can call — over a file you happened to save. If
  your analysis cannot be re-run from scratch, neither can anyone else's check of it.
- **What does the license allow?** Open does not mean unconditional. Several datasets
  below are CC BY or CC BY-SA, which means attribution and, in the ShareAlike case,
  constraints on redistribution. Some benchmarks are explicitly for evaluation only.
- **Are the units and provenance documented?** A column named `T` with no unit and no
  measurement description is a liability. Documented uncertainty is better still, and rare
  enough to be worth seeking out.

## Scoping the work

The most common failure is not a bad model but an unbounded question. A good scope is
narrow enough to finish and specific enough to be wrong: **one question, one dataset, one
comparison.**

Four patterns to avoid:

- **Predicting everything at once.** One target variable, chosen because you can say why it
  matters, beats six chosen because they were in the file.
- **Chasing size.** A 132,000-row dataset is not more informative than a 768-row one if
  your question only needs the smaller. Large data mostly buys you longer debugging cycles.
- **Skipping the baseline.** A linear model or even the column mean tells you how much of
  the signal is trivially available. Without it, a neural network's error is a number with
  nothing to compare against.
- **Splitting data that is not independent.** If measurements share a run, a batch, a
  piece of equipment, or a timestamp, a random split leaks and your validation error is
  fiction. This matters for most of the process datasets catalogued here, and it is
  covered in {doc}`Model Validation </2-regression/Topic2.2-Model_Validation>` and
  {doc}`Time Series Basics </6-advanced_topics/Topic6.1-Time_Series_Basics>`.
