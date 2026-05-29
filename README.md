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

Download or prepare the standard forecasting datasets separately. This release does not include datasets.

Expected local layout examples:

```text
E:/LTSF/all_datasets/weather/weather.csv
E:/LTSF/all_datasets/electricity/electricity.csv
E:/LTSF/all_datasets/traffic/traffic.csv
```

Adjust `--root_path` and `--data_path` in the commands below for your machine.

## Example Training Commands

Weather, horizon 96:

```bash
python run_longExp.py ^
  --is_training 1 ^
  --root_path E:/LTSF/all_datasets/weather/ ^
  --data_path weather.csv ^
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

## License

The newly added HiCR-Mamba components are released under the MIT terms in `LICENSE`. The training framework is adapted from PatchTST; please also respect the original PatchTST project's license and attribution requirements when using this code.
