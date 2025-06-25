
# Pretraining Script for MultiModalTreeVQ

This repository contains the code for pretraining a model using custom data and pretrained checkpoints.

## 📁 Directory Structure

Before running the code, please prepare the following directories and files:

### 1. `ckpt/` Directory

Create a folder named `ckpt` and place the pretrained model file inside it.

* **Download** the pretrained `.pt` file from the following Baidu Netdisk link:

  ```
  Link: https://pan.baidu.com/s/1qP-Fc9o8TBpLH_akb93Bjg
  Code: cvnb
  ```

* Place the downloaded file (e.g., `pretrain2DIM.pt`) into the `ckpt/` directory.

### 2. `raw_data/` Directory

Create a folder named `raw_data` and extract the pretraining data into it.

* **Download** the dataset from the following Baidu Netdisk link:

  ```
  Link: https://pan.baidu.com/s/1vU0fVglugG0qe8QmduKt9g
  Code: 9deg
  ```

* Unzip the `pretrain.zip` file and place its contents inside the `raw_data/` directory.
  
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


