```{contents}
:local:
:depth: 2
```

# External Resources

This book covers one path through data analytics for chemical engineers. A great deal of
excellent material covers adjacent paths, and some of it is actively maintained in ways a
static book cannot be.

## Start with CACHE

The [CACHE Corporation](https://cache.org/teaching-resources-center) Teaching Resources
Center maintains two collections that are the best single starting point for this material
in a chemical engineering context. Both are curated, annotated with audience and
prerequisite level, and updated as things change — treat them as the superset of anything
listed on this page:

- [**Machine Learning and Artificial Intelligence**](https://cache.org/teaching-resources-center/aiml)
  — course models that can be adopted wholesale, assignment ideas with defined student
  deliverables, learner pathways, datasets, and guidance on generative AI and agents. Each
  entry is tagged with best use, audience, level, expected time, access terms, and the
  limitations to know about before assigning it.
- [**Data, Statistics, and Analytics**](https://cache.org/teaching-resources-center/statistics)
  — syllabi from several departments, textbook recommendations, screencast libraries, and
  datasets, oriented toward the statistics and uncertainty side of the subject.

The [original notes](https://github.com/medford-group/data_analytics_ChE) that this book was
built from are listed on both pages, so you may well have arrived here from one of them.

## Courses and books

Course materials in this area are unusually open — most of the entries below include
lectures, notebooks, and assignments you can work through independently.

| Resource | What it is | Level |
|---|---|---|
| [Data Science and Machine Learning in Chemical Engineering](https://kitchingroup.cheme.cmu.edu/s26-06642/syllabus.html) (Carnegie Mellon) | A full course: syllabus, lectures, class exercises, 13 homework sets, and a course project | Intermediate |
| [Machine Learning for Engineers](https://apmonitor.com/pds/) (APMonitor / BYU) | Engineering-first ML course with Python notebooks, projects, and case studies; code MIT licensed | Introductory to intermediate |
| [Data-Driven Engineering](https://apmonitor.com/dde/) (APMonitor / BYU) | Sensor data, cleaning, statistics, visualization, time series, and uncertainty, with a group project | Introductory to intermediate |
| [Machine Learning in Chemical Engineering](https://edgarsmdn.github.io/MLCE_book/intro.html) | A Jupyter Book with exercises on properties, vapor–liquid equilibrium, CSTR models, control, and PCA | Intermediate |
| [Statistics for Chemical Engineers](https://github.com/vzavala/StatsBook) | Companion materials to the book: slides, scripts, worked examples, and data files | Intermediate |
| [Advanced Mathematics and Computation for Chemical Engineers](https://github.com/zavalab/Tutorials) (Wisconsin) | Tutorials with Colab links on neural networks, graph neural networks, topological data analysis, and t-SNE | Advanced |
| [AI for Chemical Engineers](https://github.com/KaihangShi/AI-for-Chemical-Engineers_UB) (Buffalo) | Slides and tutorials on fault detection, materials, optimization, and generative models; CC BY 4.0 | Intermediate to advanced |
| [Data-Driven Science and Engineering](https://www.databookuw.com/) (Brunton and Kutz) | Videos, exercises, and code on dynamics, control, and reduced-order models. Companion materials open; textbook sold separately | Advanced undergraduate to graduate |
| [Deep Learning for Molecules and Materials](https://dmol.pub/) | Interactive text with Colab notebooks on representations, graph neural networks, equivariance, and generative models; CC BY-NC 3.0 | Advanced |
| [Physics-Based Deep Learning](https://physicsbaseddeeplearning.org/) | Digital book on physical constraints, differentiable simulation, and surrogates. A GPU helps | Advanced |
| [Physics-Informed Machine Learning](https://composites.uw.edu/AI/) (Washington) | Slides and code for heat transfer, reaction discovery, and manufacturing applications | Upper undergraduate to graduate |
| [PyCSE](https://kitchingroup.cheme.cmu.edu/pycse/book/intro.html) (Kitchin) | Scientific Python for engineers, from the ground up. Strongly recommended if you are still finding your footing in Python | Introductory |

## Tools and tutorials

| Resource | Use |
|---|---|
| [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) | A thorough, exercise-driven tour of the library this book uses most |
| [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) | Regression, classification, splitting, and evaluation from first principles |
| [PySINDy examples](https://pysindy.readthedocs.io/en/latest/examples/index.html) | Discovering governing equations from dynamic data |
| [IDAES surrogate modeling examples](https://idaes-examples.readthedocs.io/en/stable/docs/surrogates/index.html) | Surrogate models inside flowsheet optimization, from DOE and NETL |
| [Streamlit](https://streamlit.io/) | Turning a finished model or analysis into a small interactive application |
| [LearnChE statistics screencasts](http://www.learncheme.com/screencasts/statistics) (Colorado Boulder) | Sixty-plus short videos on statistics topics, including software tutorials |
| [Virtual Laboratories in Probability and Statistics](http://www.math.uah.edu/stat/) | Interactive demonstrations of distributions and statistical concepts |

## Generative AI, language models, and agents

{doc}`Accessing Data with AI Tools </4-data_management/Topic4.3-Accessing_Data_with_AI_Tools>`
covers the practical use of these tools for getting at data. The resources below go deeper
into how they work and how to build with them.

| Resource | What it covers |
|---|---|
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Transformers, tokenizers, fine-tuning, and reasoning models |
| [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | Function calling, retrieval, frameworks, observability, and evaluation |
| [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | Chaining, routing, orchestration, and evaluator loops. Vendor-authored — pair it with your own testing |
| [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) | Building a language model end to end. Assumes strong ML and systems background |
| [Using LLMs in Scientific Research](https://kitchingroup.cheme.cmu.edu/s26-06642/optional/llms-for-research.html) (Carnegie Mellon) | Literature work, data analysis, coding, and keeping work repeatable |
| [Inspect AI](https://inspect.aisi.org.uk/) (UK AI Security Institute) | Building fixed test sets and clear scoring for model and agent evaluation |

For instructors deciding on course policy,
[Incorporating Generative AI in the Chemical Engineering Classroom](https://che.engin.umich.edu/2026/06/23/incorporating-generative-ai-in-the-chemical-engineering-classroom-a-perspective-for-cache-by-rebecca-k-lindsey/)
(written for CACHE) and
[Cornell Engineering's guidance](https://mtei.engineering.cornell.edu/teaching-resources/guidance-genai/)
both give concrete, adaptable starting points.

## Using any of this responsibly

The failure modes here are not exotic; they are the ordinary ones, and they are easy to
avoid on purpose and hard to avoid by accident.

- **Record where the data came from.** Source, license, units, missing-value convention, how
  it was sampled, what you changed, and the measurement uncertainty if it is stated. This
  costs ten minutes at the start and saves the entire analysis later.
- **Match the validation split to the real use.** Keep related times, batches, and equipment
  out of both training and test sets. If the model will be applied to future data, validate
  on future data.
- **Apply engineering checks a metric will not catch.** Do the predictions conserve mass and
  energy? Are the units right? Do limiting cases behave? What happens outside the training
  range? A model can have excellent cross-validated error and still predict negative
  concentrations.
- **Report how it fails, not only how well it works.** Uncertainty, failure cases, drift over
  time, and the conditions under which the model should not be used.
- **Verify what a language model tells you.** Check the calculations, the code, the
  quotations, and the sources. Confident phrasing is not evidence, and these tools are most
  persuasive exactly where they are least reliable — plausible-looking citations, and
  algebra that is formatted correctly but wrong.
- **Constrain agents that take actions.** Limit tools and permissions, protect private data,
  log what was done, set cost and time limits, and require human approval for anything
  affecting safety, equipment, or records.

For a formal framework, the
[NIST AI Risk Management Framework Generative AI Profile](https://doi.org/10.6028/NIST.AI.600-1)
sets out risk registers, test plans, and source records in a way that maps onto engineering
practice.

:::{note}
External links were last reviewed in August 2026. The CACHE pages linked at the top of this
page are maintained continuously; if something here has gone stale, check there first. If you
find a broken link, please
[open an issue](https://github.com/medford-group/DA4CHE/issues).
:::
