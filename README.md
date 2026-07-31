# <<TITLE>>

This repository contains the source code associated with ["<<TITLE>>"](<<FORTHCOMING>>), our submission to the [ARC Prize 2026](<<FORTHCOMING>>) competition.

### Repository Organization

* **[`data/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/data)** contains all our synthetic training data for experimentation.
* **[`figures/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/figures)** contains all our figures in raw format.
* **[`inc/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/inc)** contains our version of the [SSP space library](https://github.com/ctn-waterloo/sspspace).
* **[`models/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/models)** contains all our trained NNs for experimentation.
* **[`paper/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/paper)** contains our paper.
* **[`results/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/results)** contains our results data and our visualizer script, which computes all our results reported in our paper.
* **[`src/`](https://github.com/ijoffe/ARC-VSA-2026/tree/main/src)** contains all our code to implement our solver. Our code was submitted to the Kaggle competition [here](<<FORTHCOMING>>).
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
    title = {<<TITLE>>}, 
    booktitle = {20th Conference on Neurosymbolic Learning and Reasoning},
    volume = {284},
    number = {<<FORTHCOMING>>},
    pages = {<<FORTHCOMING>>},
    publisher = {Proceedings of Machine Learning Research},
    url = {<<FORTHCOMING>>},
    pdf = {<<FORTHCOMING>>},
    abstract = {<<FORTHCOMING>>},
}
```
