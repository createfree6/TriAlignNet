```markdown
# TriAlignNet

## TriAlignNet: A Triple-Path Cross-Modality Alignment Framework for Multimodal Time Series Forecasting

This repository provides the official implementation of **TriAlignNet**, a triple-path cross-modality alignment framework for multimodal time series forecasting. TriAlignNet improves forecasting by jointly modeling numerical time-series signals and their paired textual information.

The core idea is to progressively align heterogeneous modalities at three complementary levels:

1. **Distribution-level alignment** reduces the statistical discrepancy between numerical and textual representations using Maximum Mean Discrepancy (MMD).
2. **Semantic-level alignment** introduces a shared learnable anchor space to establish a stable semantic reference between modalities.
3. **Interaction-level fusion** performs fine-grained time-text interaction to integrate useful contextual information for forecasting.

---

## Overall Architecture

The overall architecture of TriAlignNet is shown below. The model employs a three-level alignment strategy to integrate information from numerical and textual modalities. Specifically, numerical features are encoded by an MLP-based numerical encoder, textual features are encoded by a pretrained language model, distribution-level alignment is achieved through MMD loss, a shared learnable anchor matrix is introduced to align multimodal features at the semantic level, and multimodal feature fusion is finally performed through time-text interaction and similarity-based retrieval.

<img width="4631" height="3481" alt="TriAlignNet" src="https://github.com/user-attachments/assets/9afdffa1-1b9a-4aae-aea2-32470c10fa61" />


---

## Repository Structure

```text
TriAlignNet/
├── data/                  # Dataset files in CSV format
├── data_provider/          # Data loading and preprocessing utilities
├── exp/                   # Experiment classes
├── layers/                # Model layers and alignment modules
├── models/                # TriAlignNet and baseline model definitions
├── scripts/               # Shell scripts for running experiments
├── utils/                 # Training, evaluation, and utility functions
├── checkpoints/           # Saved model checkpoints
├── results/               # Forecasting results
├── test_results/          # Test outputs and visualizations
├── run.py                 # Main training and evaluation entry
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Datasets

We evaluate TriAlignNet on 11 multimodal time-series datasets from different domains:

- Agriculture
- Climate
- Economy
- Energy
- Environment
- Health
- Security
- Traffic
- weather_hs
- weather_ny
- weather_sf

Each dataset contains the target numerical time series and paired textual information. The CSV files are expected to be placed under:

```text
./data/
```

A typical data file contains numerical variables, the prediction target `OT`, date information, and textual fields such as historical facts and generated textual descriptions.

Example columns may include:

```text
date, OT, start_date, end_date, fact, preds
```

You can download the datasets from Google Drive:

[Google Drive Dataset Link](https://drive.google.com/drive/folders/1KCG503FllsoSFHn7IaolrrYZ5NQRggpx?usp=sharing)

After downloading, create a folder named `data` and put all CSV files into this directory:

```bash
mkdir -p ./data
```

The expected structure is:

```text
TriAlignNet/
└── data/
    ├── Agriculture.csv
    ├── Climate.csv
    ├── Economy.csv
    ├── Energy.csv
    ├── Environment.csv
    ├── Health.csv
    ├── Security.csv
    ├── Traffic.csv
    ├── weather_hs_4hours.csv
    ├── weather_ny_4hours.csv
    └── weather_sf_4hours.csv
```

---

## Environment Setup

We recommend using a Linux environment with CUDA support.

The experiments in the paper were implemented with PyTorch and conducted on NVIDIA RTX 4090D and RTX 3060 GPUs.

Create a conda environment:

```bash
conda create -n trialignet python=3.11
conda activate trialignet
```

Install PyTorch. For CUDA 12.1, for example:

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

If you prefer a minimal installation, the following packages are commonly required:

```bash
pip install pandas scikit-learn numpy tqdm matplotlib sktime reformer_pytorch transformers accelerate
```

---

## Quick Start

A simple example can be launched using the provided script:

```bash
chmod +x ./scripts/main_forecast.sh
sh ./scripts/main_forecast.sh
```

The script runs TriAlignNet with the default configuration. You can modify the dataset list, prediction horizons, random seeds, and GPU index inside the script.

---

## Example Training Command

The following command runs TriAlignNet on the Economy dataset with input length 24 and prediction length 6:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path Economy.csv \
  --model_id Economy_seed1111_sl24_pl6 \
  --model TriAlignNet \
  --data custom \
  --features M \
  --target OT \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 6 \
  --text_emb 8 \
  --des Exp \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --d_model 512 \
  --n_heads 2 \
  --dropout 0.2 \
  --e_layers 1 \
  --seed 1111 \
  --prior_weight 0 \
  --llm_model GPT2 \
  --train_epochs 100 \
  --patience 10
```

For models that require gated HuggingFace access, please set your token as an environment variable instead of hard-coding it in scripts:

```bash
export HF_TOKEN=your_huggingface_token
```

Please do not commit private tokens to a public repository.

---

## Reproducing Main Results

### Forecasting Setting

For the main experiments, we use:

```text
Input length: 24
Prediction horizons: 6, 12, 18, 24
Metrics: MSE and MAE
```

For each dataset and each model, we evaluate the four prediction horizons:

```bash
pred_lengths=(6 12 18 24)
```

The main dataset list is:

```bash
datasets=(
  Agriculture
  Climate
  Economy
  Energy
  Environment
  Health
  Security
  Traffic
  weather_hs_4hours
  weather_ny_4hours
  weather_sf_4hours
)
```

---

## Random Seed Protocol and Repeated-Run Stability

To distinguish horizon averaging from repeated-run stability, we use the following reporting protocol.

For the main numerical-baseline comparison and the multimodal-baseline comparison, each model is independently evaluated with three random seeds:

```bash
seeds=(1110 1111 1112)
```

For each random seed, MSE and MAE are first averaged over the four prediction horizons:

```text
T = {6, 12, 18, 24}
```

Then, the final results are reported as:

```text
mean ± std
```

where the mean and sample standard deviation are computed across the three independent random seeds.

In other words:

1. Run each model with seed 1110, 1111, and 1112.
2. For each seed, average the results over the four horizons.
3. Compute mean and sample standard deviation across the three seed-level averages.

Unless otherwise specified, auxiliary experiments such as ablation studies, sensitivity analyses, robustness evaluations, and visualization experiments use the fixed random seed:

```bash
seed=1111
```

This setting follows the original experimental configuration while keeping the main comparison tables supported by repeated-run statistics.

---

## Main Result of Numerical and Multimodal Forecasting

### Numerical-Baseline Comparison

Table 1 reports the main forecasting results on eleven datasets. Lower MSE and MAE indicate better performance. Results are averaged over four forecasting horizons with input sequence length 24. In the revised evaluation protocol, the main comparison reports mean and standard deviation across repeated random seeds.

<img width="2474" height="1747" alt="Single-modal main experimental results" src="https://github.com/user-attachments/assets/607ad41c-8110-4a72-8a32-389a1b47af84" />

### Multimodal-Baseline Comparison

Table 2 compares TriAlignNet with representative multimodal forecasting baselines. All models are evaluated under the same input length and prediction horizons. The reported values summarize the performance over prediction horizons and repeated random seeds.

<img width="2481" height="2931" alt="Multimodal baseline detailed results" src="https://github.com/user-attachments/assets/d04bca29-1b97-483a-8a1e-72f8300073ac" />

---

## Running All Main Experiments

You can modify `./scripts/main_forecast.sh` to reproduce the full main results.

A typical configuration is:

```bash
all_models=("TriAlignNet")
GPU=0
root_path=./data

seeds=(1110 1111 1112)

datasets=(
  Agriculture
  Climate
  Economy
  Energy
  Environment
  Health
  Security
  Traffic
  weather_hs_4hours
  weather_ny_4hours
  weather_sf_4hours
)

pred_lengths=(6 12 18 24)
```

Then run:

```bash
chmod +x ./scripts/main_forecast.sh
sh ./scripts/main_forecast.sh
```

The experimental logs and results are saved to the output files specified by the `--save_name` argument and the default result directories.

---

## Key Hyperparameters

The main experiments use the following configuration unless otherwise specified:

| Hyperparameter | Value |
|---|---:|
| Input length | 24 |
| Label length | 12 |
| Prediction lengths | 6, 12, 18, 24 |
| Hidden dimension | 512 |
| Number of attention heads | 2 |
| Number of encoder layers | 1 |
| Dropout | 0.2 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Training epochs | 100 |
| Early-stopping patience | 10 |
| LLM encoder | GPT-2 |
| Main random seeds | 1110, 1111, 1112 |
| Auxiliary-experiment seed | 1111 |

---

## Output Files

During training and evaluation, the code will generate:

```text
checkpoints/       # model checkpoints
results/           # forecasting metrics
test_results/      # test predictions and visual outputs
```

The exact result file name can be controlled with:

```bash
--save_name result_longterm_forecast
```

A typical result record includes:

```text
mse: ...
mae: ...
rmse: ...
mape: ...
mspe: ...
```

---

## Notes on Textual Modality

TriAlignNet uses textual information paired with the numerical time series. The text modality can include historical facts, textual summaries, or future-related descriptions, depending on the dataset construction.

The textual modality is encoded by a pretrained language model. In the default setting, we use:

```bash
--llm_model GPT2
```

Other LLM backbones can also be specified through the `--llm_model` argument if the corresponding model and tokenizer are supported.

---

## Reproducibility Checklist

To reproduce the main results, please check the following:

- All datasets are placed under `./data/`.
- The input length is set to 24.
- The prediction horizons are set to 6, 12, 18, and 24.
- The main tables are evaluated with seeds 1110, 1111, and 1112.
- The reported `mean ± std` values are computed across seed-level averages.
- The same textual representation setting is used across compared models.
- Auxiliary experiments use seed 1111 unless otherwise specified.
- Private tokens or machine-specific paths are not hard-coded in public scripts.

---

## Acknowledgement

We sincerely appreciate the following repositories for their valuable codebases and datasets:

- TimeCMA: https://github.com/ChenxiLiu-HNU/TimeCMA
- DMMV: https://github.com/D2I-Group/dmmv
- FreqLLM: https://github.com/biya0105/FreqLLM

We also thank the open-source time-series forecasting community for providing useful implementations and benchmarks.

---

## Citation

If you find this repository useful for your research, please consider citing our work:

```bibtex
@article{TriAlignNet2026,
  title   = {TriAlignNet: A Triple-Path Cross-Modality Alignment Framework for Multimodal Time Series Forecasting},
  author  = {Ye, Junjie and Zhao, Chunna},
  journal = {Preprint},
  year    = {2026}
}
```

The BibTeX entry will be updated after publication.

---

## Contact

If you have any questions or concerns, please contact us by email or submit an issue:

- Junjie Ye: yejunjie@stu.yun.edu.cn
- Chunna Zhao: zhaochunna@ynu.edu.cn
```
