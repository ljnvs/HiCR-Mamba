# Datasets

This directory contains gzip-compressed copies of the public forecasting benchmark files used by the HiCR-Mamba experiments.

Included files:

- `weather/weather.csv.gz`
- `electricity/electricity.csv.gz`
- `traffic/traffic.csv.gz`
- `ETT-small/ETTh1.csv.gz`
- `ETT-small/ETTh2.csv.gz`
- `ETT-small/ETTm1.csv.gz`
- `ETT-small/ETTm2.csv.gz`

The files are compressed because `traffic.csv` is larger than GitHub's ordinary single-file limit when stored uncompressed. The training code reads these files through `pandas.read_csv`, which supports `.csv.gz` directly. For example:

```bash
python run_longExp.py --root_path datasets/weather/ --data_path weather.csv.gz --data custom --features M --enc_in 21 --dec_in 21 --c_out 21 --model HiCRMamba --pm_variant hicr
```

If a local workflow expects plain CSV files, decompress the corresponding `.csv.gz` file and keep the same directory layout.
