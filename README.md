# Student Engagement Prediction

A project focusing on predicting student engagement through the analysis of body signals, employing self-attention mechanisms to enhance model performance.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Lyfpy/EngagePredictTrans.git
cd EngagePredictTrans
pip install -r reqs.txt
```

## Usage

1. Process the Openface outputs:

    ```bash
    python3 OpenFaceDataset.py
    ```

2. Train the model:

    ```bash
    python3 train_and_test.py
    ```

## Requirements

The project depends on several Python libraries listed in `reqs.txt`, including but not limited to scipy, stumpy, torch, pytorch-lightning, and tsfresh.
