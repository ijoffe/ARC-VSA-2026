# Neurosymbolic Program Synthesis for the Abstraction and Reasoning Corpus using Vector Symbolic Algebras

This repository contains the source code associated with ["Neurosymbolic Program Synthesis for the Abstraction and Reasoning Corpus using Vector Symbolic Algebras"](https://openreview.net/pdf?id=aNWQsETQnt), our submission to the [ARC Prize 2026](https://arcprize.org/competitions/2026) competition.

### Repository Organization

* **[`data/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/data)** contains all our synthetic training data for experimentation.
* **[`figures/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/figures)** contains all our figures in raw format.
* **[`inc/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/inc)** contains our version of the [SSP space library](https://github.com/ctn-waterloo/sspspace).
* **[`models/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/models)** contains all our trained NNs for experimentation.
* **[`paper/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/paper)** contains our paper.
* **[`results/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/results)** contains our results data and our visualizer script, which computes all our results reported in our paper.
* **[`src/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/src)** contains all our code to implement our solver. Our code was submitted to the Kaggle competition [here](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/neurosymbolic-program-synthesis-vsa).
* **[`tasks/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/tasks)** contains all ARC-AGI-1 and ARC-AGI-2 tasks for testing.

### Usage

First, set up an execution environment:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Second, run the interactive `.ipynb` notebooks as-is to collect and visualize results.

### Citation

If you use our code or paper, please cite our work as:

```
@inproceedings{joffe2026arcvsa,
    author = {Isaac Joffe and Chris Eliasmith},
    year = {2026},
    title = {Neurosymbolic Program Synthesis for the {Abstraction and Reasoning Corpus} using Vector Symbolic Algebras}, 
    booktitle = {20th Conference on Neurosymbolic Learning and Reasoning},
    publisher = {Proceedings of Machine Learning Research},
    url = {https://openreview.net/pdf?id=aNWQsETQnt},
    abstract = {The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is a popular artificial intelligence (AI) benchmark that tests fluid reasoning. Humans solve ARC-AGI with ease, but it is extremely difficult for all but the most advanced AI systems. In this paper, we propose a novel, neurosymbolic ARC-AGI solver with cognitive inspiration from dual process theory and vector symbolic algebra (VSA) research. Our solver structures solutions to ARC-AGI tasks as object-centric programs implemented in a bespoke DSL, and represents these programs using VSAs. Additionally, our solver synthesizes solutions to ARC-AGI tasks by integrating neural System 1 intuitions with procedural System 2 reasoning, and implements both using VSAs. Quantitatively, our solver scores 13.5 percent on ARC-AGI-1-Train and 3.5 percent on ARC-AGI-1-Eval, indicating success; qualitatively, our solver is efficient and interpretable, outperforming naive methods by 7 times and brute force methods by 235 times. Our code is available at: https://github.com/ijoffe/ARC-VSA-2026.},
}
```
