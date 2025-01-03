## Training
Each model should expose a generic method `fit(self, x, y, **kwargs)` where (x, y) can be one of these supported cases:

1. data directories
    - x : Path-like object, directory storing individual <model_input> identified by file name
    - y : Path-like object, directory storing target json output in standard format, with matching file names with <model_input> directory.
2. lists of paths
    - x : list containing paths to individual <model_input>
    - y : list of paths to individual json labels
3. x and y are iterators containing data
    - x : iterator containing pre-loaded <model_input>
    - y : iterator containing pre-loaded targets in standard format

Fit should only concern itself with fitting the given data, not with calculating accuracies, visualization or performing evaluation on a different dataset. In cases where the authors wish to evaluate during a `fit` routine, it is possible to do so only if `evaluate` itself does not modify internal states of the model.

Each model should expose additional methods for common training tasks:

- `save(self, weights_path=None)` : save current checkpoint to disk. If None, defaults to a naming scheme defined by the model author
- `load(self, weights_path=None)` : load checkpoint to continue training from, or load saved weights for inference. If None, auto-discover the latest checkpoint in current directory to load
- `export(self, weights_path=None)` : export weights for inference only. If None, defaults to a naming scheme defined by the model author


## 1. Input Format:

(Example)
A single numpy array with the following properties:

- type: list of numpy.ndarray
- dtype: uint8
- value range: 0 - 255 inclusive
- shape: (height, width, 3)

Example:

```python
inputs = [array([[[0, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0]]], dtype=uint8),
          array([[[0, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0]]], dtype=uint8)]   
```


## 2. Output format


- type: list of dicts
- each dict has these key value pairs:
    - `text` : UTF-8 encoded str
    - `confidence_by_character` : list of floats from 0.0 - 1.0 inclusive
    - `confidence_by_field` : float from 0.0 - 1.0 inclusive

Example:

```python
result = [{
              'confidence_by_character': [0.99, 0.99],
              'confidence_by_field': 0.99,
              'text': '平成',
          }, 
          {
              'confidence_by_character': [0.99, 0.99],
              'confidence_by_field': 0.99,
              'text': '令和',
          }]
```

## 3. Usage:

We'll take a model as an example. To get the metrics result, we can run:
```python
from <theme> import ModelName

model = ModelName(**kwargs)
result = model.fit(x, y, **kwargs)

```
