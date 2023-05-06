# Discrete Spider monke optimization for TSP problem
#### [Implementation of discrete spider monkey optimization for traveling salesman problem](https://www.youtube.com/watch?v=1XwimHTXYdQ).

<p align="center">
    <img src="https://user-images.githubusercontent.com/44975359/236620465-8a519905-4af2-44e9-8c12-ba91c26f6860.png" width="250" height="250" style="display: block; margin: 0 auto"/>
</p>


## Usage
Run optimization
```
python monke.py PROBLEM_PATH --config-path=config/optmizer_config.yaml --n-iter=10000 --timeout-seconds=300
```

## Help
```
Usage: python monke.py [OPTIONS] PROBLEM_PATH

  Optimize traveling salesman problem using spider-monkey discrete optimization algorithm.

Arguments:
  PROBLEM_PATH  Path to problem.  [required]

Options:
  --config-path TEXT              Path to configuration of optimizer. [default: config/optmizer_config.yaml]
  --n-iter INTEGER                Number of iterations to perform.  [default: 10000]
  --timeout-seconds INTEGER       Maximum optimization time  [default: 300]
```

## Sources
* [Algorithm](https://www.sciencedirect.com/science/article/pii/S1568494619306684)
* [Data](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/)
* [Known solutions](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ATSP.html)