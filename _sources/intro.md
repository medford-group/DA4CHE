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

```{contents}
:local:
:depth: 3
```

# Introduction

This Jupterbook is derived from a collection of notes, originally developed as a part of the "Data Analytics for Chemical Engineers" course at Georgia Tech (ChBE 4745/6745). The [original notes](https://github.com/medford-group/data_analytics_ChE) have been re-organized into this Jupyterbook format to make it easier to navigate, and in the process some re-organization and refining was done. Several AI assistant tools were used in this conversion process, and while all content has been manually reviewed, some artifacts may still be present. At present, this version is still in "beta testing", and the transition is not complete. Hopefully, the full transition will be done by Fall 2026. If you find any issues with these notes, including significant discrepancies from the original version or just suggestions for improvements, please open an "issue" on the Github page (preferred) or email [A.J. Medford](mailto:ajm@gatech.edu).

## How to use this book
If you are a student in ChBE 4745/6745, you should think of this as a supplement to the [original notes](https://github.com/medford-group/data_analytics_ChE), which are still referenced in the lecture videos. This "book" form has been updated to include more content, clean up some issues, and modernize some of the code to follow current standard practices. 

You can navigate the book using the left-hand table of contents for modules, and within each module you will find a dynamic table of contents on the right-hand side. You can download any page of this book as a Jupyter Notebook (`.ipynb`) using the links in the upper right and run them locally. You can also download the [entire source repo](https://github.com/medford-group/DA4CHE) for this book, although the source files are now in MyST Markdown and will need to be converted using `jupytext`. There is an `environment.yml` file in the Github repo for this book which you can use to create a Conda environment with all necessary dependencies (`conda env create -n DA4CHE -f environment.yml`). The LLMs are generally quite knowledgeable about these technicalities, and may be useful in helping you get up and running.

If you are not a student in ChBE 4745/6745, congrats on finding this unpublicized resource. You are welcome to use it as you see fit and I hope you find it helpful. Feedback is always welcome and appreciated.

If you are an instructor of another similar course, or want to use these materials for educational purposes, please feel free to use and modify any of these resources as you see fit for your course. The book is available under the MIT license. Attribution is greatly appreciated, and if you find the book particularly useful or have any questions about adapting it please send a note to [A.J. Medford](mailto:ajm@gatech.edu).

## Current status

At the time of this update, only the "Numerical Methods" and "Regression" modules have been converted. This should be treated as an "alpha version", and is subject to change over the coming year as additional modules are converted.

# Overview of Contents

## {doc}`Numerical Methods <1-numerical_methods/intro.md>

## {doc}`Regression <2-regression/intro.md>

## Classification - Coming Soon!

## Data Management - Coming Soon!

## Advanced Topics - Coming at some point...

# Attributions

These notes have been collected, refined, and revised numerous times over the years, and it is impossible to directly attribute all ideas and notes to their original sources. Links or references are included in the notes where ideas or quotes are taken directly. I want to extend a special thanks to Leo Chiang and Ivan Castillo at Dow Chemical who provided the chemical process dataset used as an example throughout the course. At the time of creation, there were few if any similar datasets that were publicly available. I also want to give credit to the many fantastic resources and blog posts from Prof. John Kitchin, who is a pioneer of Python and data science education in chemical engineering, and I strongly recommend his ["PyCSE" book](https://kitchingroup.cheme.cmu.edu/pycse/book/intro.html#) for anyone (especially engineers) who are just getting started with Python. In addition, I am grateful to the general openness of the machine learning community, which has provided access to both code and helpful blog posts and other learning resources -- the content created by Sebastian Raschka was particularly useful in creating the first version of the course. The course textbooks, "Machine Learning Refined" and "Elements of Statistical Learning" were heavily influential when crafting the original version of the notes. I also want to thank colleagues who have adapted these resources and provided valuable feedback along the way, especially Prof. Chris Paolucci at UVA. I must als acknowledge that LLMs, including Claude and ChatGPT, were particularly useful when refining the notes, both by helping with menial tasks to set up the Jupterbook, and in providing additional suggestions of exercises and elaboration on the notes.

Finally, I would like to acknowledge the people who have helped create and support the ["Data Science for the Chemical Industry"](https://www.chbe.gatech.edu/online-graduate-certificate-data-science-chemical-industry) certificate program, which has made it possible to create this course and book in the form it currently is. This includes Profs. David Sholl, Carson Meredith, and Martha Grover who were instrumental in starting the program, and Prof. Fani Boukouvala who has provided endless feedback and support along the way (and teaches the companion course, ChBE 4745/6745 on "Data Driven Process Systems Engineering"). Support from Georgia Tech Professional Education was also critical, especially from Fatimah Wirth. I am grateful to be a part of this program, and hope that this book grows along with it.
