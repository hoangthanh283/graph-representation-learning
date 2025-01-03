
<div align="center">
    <h1> Model Training Template </h1>
    <!-- Model philosophy, change it to anything you want -->
    <i> With great model comes great training manual. </i>
</div>

---
#### 1. [Model Overview](#overview)
#### 2. [Install Environment](#environment)
#### 3. [Dataset Preparation](#data)
#### 4. [Training Configuration](#configuration)
#### 5. [Start Training](#training)
#### 6. [Monitor Training Session](#monitor)
#### 7. [Note and Insight Knowledge](#note)  
#### 8. [FAQ](#faq)  

---
## <a id="overview">1. Model Overview </a>
Provide a quick introduction about the model.

In this section: 
+ An overview picture for the model architecture. 
+ A brief summary about the model.
+ All reference documents (links to papers, reports, related works, ...).
---

## <a id="environment">2. Install Environment </a>
Steps to create environment and install dependencies to train this model (on local machine).

---

## <a id="data">3. Dataset Preparation </a>
Data preparation before training.

### 3.1 Download Dataset
Script or quick command to download & process a simple data for training.

In this section: 
+ User able to quickly get the data without further investigation data.
+ Contain simple link to download data from Box.
+ If getting the data is complicated, then must contain references to use it. (Ex: the data obtained via Data Management system)

### 3.2 Generate Dataset (optional)
Step-to-step to generate fake data for training.

### 3.3 Preprocess Dataset (optional)
Steps to pre-processing downloaded data (if it is in raw format, e.g. data from Datapile -> preprocess -> masked data).

Nice-to-have:
+ Feature to testing/sanitize input data before training. Example: `$pytest tests/<my_model>/test_io.py`


---

## <a id="configuration">4. Training Configuration </a>
This section contain all necessary configuration which a user need to know.

In this section: 
+ All necessary configs for training session
+ Sort important/frequent configs first
+ Include configs name, its describe, side note about insights how to change that config effectively.
+ Prefer to split configures into 3 below categorizes: training, architecture, data

### **4.1 Training configuration**  
How to configure training session, ex: batch_size, n_epochs, ...

### **4.2 Data configuration**   
How to configure data flow, ex: scale_factor, binarize, grayscale, ...


### **4.3 Architecture configuration**
How to configure model architecture, ex: n_embedding, n_layers, , ...

---

## <a id="training">5. Start Training </a>
How to start training on local machine (not include SageMaker in here, as it would be another document)

+ Script or command to run training model.
+ Where is logs, checkpoints located? 
+ Keep it short and simple as possible.

---

## <a id="monitor">6. Monitor Training Session </a>
Logging output and monitor training process:
+ What's printing on the console? (loss, metric, ...)
+ Tensorboard description, explanation (optional)
+ Warning, note for abnormal behaviours (optional, ex: loss is nan then terminating process and retrain with parameter ...; metric AC3 negative is okay)


---

## <a id="note">7. Note and Insight </a>
Further insight knowledge and note to effectively fine-tune this model.

Examples:

### 7.1 Case 1: Data contains many skew lines
In case of there are many skew lines on the data, we suggest to add Toyota4 data for fine-tuning also. As, Toyota4 data has many same skew data like this, training on it help model more 
### 7.2 Case 2: There are very few data on this project
You can tackle this problem by generating fake data (read Data Preparation above), or adding on-the-fly augmentation.
### 7.3 Case 3: Loss strategy
To get the model converge faster, we should using dice loss for first 3 epochs, then using dice loss + CE loss for the rest.

---
## <a id="faq">8. FAQs </a>

Frequently Asked Questions for training this model.
 
Examples:

**Q1: How long does it take to run 1 epoch of 100 training images on T4 GPU?**

**Ans:** xxx hours

**Q2: What is interest about this model?**

**Ans:** Incredible inference speed and competitive accuracy, due to effective feature extraction. 

**Q3: I got following error when running training script:**
```
traceback or Issue ID:
...
```
**Ans:** This error is due to ..., you can solve it by ...

---
> ## Contributors
> List of maintainers and contributors who help to make this model.
> - Contributors: 
> - Maintainers: 
