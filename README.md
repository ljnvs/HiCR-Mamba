# HiCR-Mamba

HiCR-Mamba is an attention-free patch state model for high-dimensional multivariate time series forecasting. It replaces the PatchTST Transformer encoder with a Mamba-style selective patch state mixer and introduces a Channel-Recent Adapter for high-dimensional variables.

This repository is a minimal, reproducible release extracted from the experimental workspace. It keeps only the files required to train and evaluate HiCR-Mamba. The training/data framework is adapted from the PatchTST supervised codebase, while the HiCR-Mamba model components are newly added.

## Model

The released model corresponds to the full HiCR-Mamba configuration:

- `model`: `HiCRMamba`
- `pm_variant`: `hicr`
- Core modules:
  - Selective Patch State Mixer
  - Channel-Controlled State Modulation
  - Gated Recent-State Enhancement

For component-level checks, `pm_variant=base` uses only the plain selective patch state mixer, `pm_variant=channel_gate` enables Channel-Controlled State Modulation, and `pm_variant=recent_state` enables Gated Recent-State Enhancement.

## Directory Layout

```text
.
|-- data_provider/          # dataset loaders
|-- datasets/               # gzip-compressed benchmark files
|-- exp/                    # training/evaluation loop
|-- layers/                 # HiCR-Mamba backbone and helper layers
|-- models/                 # model wrapper
|-- utils/                  # metrics, tools, time features
|-- scripts/                # example commands
|-- run_longExp.py          # main entry
|-- requirements.txt
`-- README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The code was developed with PyTorch and tested locally on a single NVIDIA RTX 4070. Newer PyTorch versions should work, but exact numerical results may vary across CUDA/PyTorch versions.

## Data

This release includes gzip-compressed copies of the public forecasting benchmark files used in the paper:

- Weather
- Electricity
- Traffic
- ETT-small: ETTh1, ETTh2, ETTm1 and ETTm2

```text
datasets/weather/weather.csv.gz
datasets/electricity/electricity.csv.gz
datasets/traffic/traffic.csv.gz
datasets/ETT-small/ETTh1.csv.gz
datasets/ETT-small/ETTh2.csv.gz
datasets/ETT-small/ETTm1.csv.gz
datasets/ETT-small/ETTm2.csv.gz
```

The files are stored as `.csv.gz` so that large datasets such as Traffic stay below GitHub's ordinary file-size limit. The data loader uses `pandas.read_csv`, which can read gzip-compressed CSV files directly. You may therefore pass `--data_path weather.csv.gz`, or decompress the files locally and use the corresponding `.csv` names.

## Example Training Commands

Weather, horizon 96:

```bash
python run_longExp.py ^
  --is_training 1 ^
  --root_path datasets/weather/ ^
  --data_path weather.csv.gz ^
  --model_id weather_96_hicr ^
  --model HiCRMamba ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --enc_in 21 ^
  --dec_in 21 ^
  --c_out 21 ^
  --d_model 64 ^
  --d_ff 128 ^
  --e_layers 2 ^
  --patch_len 16 ^
  --stride 8 ^
  --pm_variant hicr ^
  --pm_d_state 16 ^
  --pm_expand 2 ^
  --pm_d_conv 3 ^
  --pm_bidirectional 1 ^
  --pm_residual_scale 0.5 ^
  --pm_recent_k 3 ^
  --pm_channel_rank 8 ^
  --batch_size 32 ^
  --learning_rate 0.0001 ^
  --train_epochs 20 ^
  --patience 3 ^
  --itr 1
```

Electricity uses `--enc_in 321 --dec_in 321 --c_out 321`. Traffic uses `--enc_in 862 --dec_in 862 --c_out 862`; reduce `--batch_size` if GPU memory is insufficient.

## Notes

- The implementation is attention-free in the HiCR-Mamba backbone.
- The code is based on the PatchTST supervised training framework, with a new HiCR-Mamba model added.
- Results are saved under `results/`, `test_results/`, and `checkpoints/` when training is run.

## Acknowledgements

This release adapts the training loop, data loading utilities and several helper layers from the PatchTST supervised codebase. We thank the PatchTST authors for releasing their implementation:

- PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- Official repository: https://github.com/yuqinie98/PatchTST

The HiCR-Mamba backbone and Channel-Recent Adapter are newly added in this repository.

## License

The newly added HiCR-Mamba components are released under the MIT terms in `LICENSE`. The training framework is adapted from PatchTST; please also respect the original PatchTST project's license and attribution requirements when using this code.
