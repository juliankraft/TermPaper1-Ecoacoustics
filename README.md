## Accessibility of Ecoacoustics: A Deep Learning Approach to Insect Sound Classification

### Abstract

This study aims to reproduce the results of a previous research on insect sound classification using deep learning and to assess the accessibility of this technology. Usingthe InsectSet32 dataset, which contains 335 recordings of 32 insect species, a convolutional neural network (CNN) with residual blocks was developed. The model was trained using the PyTorch framework and optimized through hyperparameter tuning. The bestperforming model achieved an accuracy of 0.706 on the validation set and 0.649 on the test set, closely aligning with the original study’s findings. This work demonstrates that, with appropriate knowledge and relatively affordable hardware, such as a regular gaming computer equipped with a GPU, non-experts can effectively utilize deep learning models for ecoacoustic applications. The study underscores the potential of combining ecoacoustics with artificial intelligence to advance biodiversity monitoring, providing a non-invasive and efficient method for large-scale environmental assessments.

**Author:**         Julian Kraft  
**Tutor:**          Dr. Matthias Nyfeler  
**Institution:**    Zurich University of Applied Sciences (ZHAW)
**Program:**        BSc Natural Resource Sciences  
**Project:**        Term Paper 1  
**Date:**           2024-07-16

**Data:** InsectSet32 dataset from Zenodo: [link](https://zenodo.org/records/7072196)

**Paper:** [link](LaTeX/main.pdf)
**Visualizations:** [link](code/visualizations.ipynb)

### Repository Content

This repository provides all the relevant code, data and training logs as well as the LaTeX source code and
and all visualizations used in the term paper.

### Repository Structure

- `LaTeX/`: LaTeX source code of the term paper
- `code/`: Python code for the CNN model and the evaluation
- `data/`: InsectSet32 dataset
- `logs/`: training logs of the CNN model

### Environment

The model was trained on a regular gaming computer equipped with an NVIDIA GeForce RTX 2060 SUPER.
The Environment was set up using Anaconda the packages used are listed in [environment.yml](environment.yml).

### Usage

The trining of the different models from the hyperparameter tuning can be started by running the [run_experiment.py](/run_experiment.py) script.