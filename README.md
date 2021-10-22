# A Probabilistic State Space Model for Joint Inference from Differential Equations and Data

## Paper

_"A Probabilistic State Space Model for Joint Inference from Differential Equations and Data"_

* arXiv: [https://arxiv.org/abs/2103.10153](https://arxiv.org/abs/2103.10153)

## Install

```
pip install -e .
```

## Run experiments

1. Change working directory to experiment folder, e.g.

```
cd exp/exp003_standard_sird
```

2. Execute experiment, e.g.

```
./run_experiment ger
```

## Plot results

(Make sure you are in the directory that contains the result directory that has just been created.)

1. Execute plotting script, e.g.

```
log-sird-plot <result-directory>
```