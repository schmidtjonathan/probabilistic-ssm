# A Probabilistic State Space Model for Joint Inference from Differential Equations and Data

### Note:

Most of what makes up the method presented in the paper is implemented in [ProbNum](http://www.probabilistic-numerics.org/en/latest/).
Since ProbNum is under development and thus subject to regular changes, the paper code has yet to be adapted in parts.


## Paper

Accepted (Poster) at NeurIPS 2021:

[NeurIPS 2021 Proceedings](https://papers.nips.cc/paper/2021/hash/6734fa703f6633ab896eecbdfad8953a-Abstract.html)
(or on [OpenReview](https://openreview.net/forum?id=7e4FLufwij) ).


Please cite this work as

```
@inproceedings{
    schmidt2021a,
    title={A Probabilistic State Space Model for Joint Inference from Differential Equations and Data},
    author={Jonathan Schmidt and Nicholas Kr{\"a}mer and Philipp Hennig},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=7e4FLufwij}
}
```


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