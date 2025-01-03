# Metrics in THEME
----
## 1. Definitions:
### a. True Positive(TP), False Positive(FP), False Negative(FN), True Negative(TN):

By setting up a threshold and applying IOU, we can then tell the predicted bounding box (detection result) is a TP sample (valid sample) or FP sample (non-valid sample).
* **TP**: A detection with **IOU $\geq$ threshold** (A correct detection)
* **FP**: A detection with **IOU $\lt$ threshold** (A wrong detection)
* **FN**: A ground truth that is not detected
* **TN**: A bounding box that is correctly not detected (Does not apply)

The **threshold** is depending on the metric. Normally it's set to 0.5, 0.75, or 0.95.


## 2. <theme> Metric:

### a. Precision:
Precision is how well a model can get **only** the relevant objects. It's the percentage of correct positive predictions.

<div style='text-align:center'><img src='./pics/precision1.png'></div>


### b. Recall:
Recall is how well a model can get **all** the relevant objects. It's the percentage of true positive detected among all relevant **ground** truths.

<div style='text-align:center'><img src='./pics/recall1.png' style='width:200px'></div>


### c. F1-score (F1-measure):
The layout analysis usually use F1-score as the measurement for model performance. It makes a balance between precision and recall.

<div style='text-align:center'><img src='./pics/f1_score1.png' style='width:200px'></div>

### d. <Other metric for this theme>:


## 3. Usage:

We'll take a model as an example. To get the metrics result, we can run:
```python
from <theme> import ModelName
from <theme>.metrics import metric_name

model = ModelName(weights_path='path/to/model.pb')
result = model.process('path/to/image.png')

metric_out = metric_name('path/to/label.json', result)
```

Please state the available label format / schema below for metric usage.


### Reference:

- (If any)

