<div align="center">

# Life Long Learning
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)]()
[![Code style: pep8](https://img.shields.io/badge/code%20style-pep8-yellow)](https://www.python.org/dev/peps/pep-0008/k)


</div>

## **Installation**

It is recommended that you create a fresh isolated Python 3.7 environment with the latest pip to prevent potential version conflicts

### **Install from the master branch**

```bash
pip install --upgrade pip
pip install -e .
```

### **Development environment**
If you want to contribute to this project, please clone this repo, install dev dependencies and pre-commit hooks:

```bash
pip install -e ".[dev]"
pip install pre-commit
pre-commit
```

### **Running tests**

#### TODO

## **Inference API**

Refer to scripts/demo_inference.py as an example to infer a trainee models under the scripts directory.

## **Training API**

In this section, we will introduce a brief guideline to quickly fine-tune models in this repo, for detailed documentation of each model please refer below:
### **Prerequisites**

  Follow these preparation steps before training.

  ```bash
  Step 1: Prepare training data
    ├── classes.json                                                                                           
    ├── corpus.json                                                                                            
    ├── train                                                                                                  
    │   ├── sample_1.json                                                                          
    │   ├── sample_2.json                                                                          
    │   ├── sample_3.json  
    │   └── ...                                                                    
    └── val                                                                                                    
        ├── sample_1.json                                                                          
        ├── sample_2.json                                                                          
        ├── sample_3.json  
        └── ... 
  ```
  Step 2: Clone config file in configs/base_config.yaml and edit it based on your task.

  Step 3: Refer to scripts/demo_traning.py as an example to train models under the scripts directory.

  ### **Training configuration**

  Important configuration of training script, for more information you can check args at `configs/base_config.yaml`

  ### **Start training**

  Refer to `scripts/demo.py` as an example to train model.

  ```shell script
  import argparse

  from gnn.cl_warper import GNNLearningWarper
  from gnn.models import GraphCNNDropEdge

  if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="CL configurations")
      parser.add_argument("--config", default=None, type=str, help="Path to the configuration file.")
      args = parser.parse_args()

      # Define model instances.
      model = GraphCNNDropEdge(input_dim=4369, output_dim=53, num_edges=6, net_size=256)

      # Define the optimzation warper.
      warper = GNNLearningWarper(model, config_path=args.config)
      warper.train()
  ```
  To visualize training progress, use TensorBoard with `tensorboard --logdir your_summarize_dir` 


## **Development**

Install dev dependencies and pre-commit hooks:

```shell script
pip install -e ".[dev]"
pip install pre-commit
pre-commit
```
Check your code against [`PEP 8`](http://www.python.org/dev/peps/pep-0008/) conventions.
