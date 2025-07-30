
# Pretraining Script for CHMR

This repository contains the code for pretraining a model using custom data and pretrained checkpoints.

## 📁 Directory Structure
### Data Preparation
Create a folder named `raw_data` and extract the pretraining data into it.
<!--
Create a folder named `raw_data` and extract the pretraining data into it.

**Download** the dataset from the following Baidu Netdisk link:

Link: https://pan.baidu.com/s/1vU0fVglugG0qe8QmduKt9g
Code: 9deg

Unzip the `pretrain.zip` file and place its contents inside the `raw_data/` directory.
-->

### Data Availability

Due to anonymity requirements during the peer review process, the dataset download link will be made publicly available after the paper is accepted.
  
---

## 🛠 Requirements

Make sure your environment has the required dependencies installed. Example:

```bash
pip install -r requirements.txt
```

---

## 🚀 Pretraining

To start the pretraining process, run the following command:

```bash
python pretrain.py \
  --model-path "ckpt/model_name.pt" \
  --lr 1e-4 \
  --wdecay 1e-8 \
  --epoch 300 \
  --batch-size 3072
```

### Parameters

* `--model-path`: Path to save the pretrained model (.pt file)
* `--lr`: Learning rate
* `--wdecay`: Weight decay
* `--epoch`: Number of training epochs
* `--batch-size`: Training batch size


