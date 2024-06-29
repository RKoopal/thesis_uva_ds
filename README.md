# Project

This repository contains some of the code used in the thesis research titled: 'Is Merging All You Need?: Creating stronger multilingual multi-task LLMs using model merging.' Written by Reimer Sjouke Theodoor Koopal and submitted on 30-06-2024 in partial fulfillment for the Msc Information Studies - Data Science, at the University of Amsterdam. The research was commissioned by Deloitte Nederland as a thesis internship project.

## DISCLAIMER

This code is incomplete and will not run on its own. Part of the codebase used, including most of the training, testing, and evaluation scripts, cannot be shared due to privacy concerns and are therefore 'hidden'. This repository merely serves as an example to illustrate how the research and evaluation were performed.

## Databricks

All final experiments whose results were used in the paper were performed on the Databricks platform. The 'databricks' folder contains the scripts created to run the code on the Databricks platform. Note that this folder is created to differentiate these files and does not reflect the directory structure used on the Databricks platform.

## imported.ipynb

Functions that were added to the 'hidden' code for this research are located in `databricks/imported.py`. This file did not exist during the research; it simply holds functions that were part of a larger script that cannot be shown here.

## FlagEmbedding contribution

For this research, it was necessary to add functionality to the LM-Cocktail library, which is part of the FlagEmbedding repository. The Pull Request and Issue that led to this contribution are linked below.

- PR: [https://github.com/FlagOpen/FlagEmbedding/pull/761](https://github.com/FlagOpen/FlagEmbedding/pull/761)
- Issue: [https://github.com/FlagOpen/FlagEmbedding/issues/750](https://github.com/FlagOpen/FlagEmbedding/issues/750)

