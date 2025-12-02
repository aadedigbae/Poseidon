# EDA Summary (00_data_understanding_eda)

## Datasets
- **WQD.xlsx** raw: (4300, 15), canonical: (4300, 7)
- **Monteria_Aquaculture_Data.xlsx** raw: (4345, 7), canonical: (4345, 7)

## Canonicalization
- **WQD**: temperature, pH, **turbidity_cm** → **turbidity_proxy = 1/cm**, do, ammonia, water_quality
- **Monteria**: timestamp, temperature, pH, **turbidity (NTU)**, do

## Key Outputs
- Histograms & correlations in **/images/**
- Interim clean CSVs in **/data/interim/**:
  - wqd_clean.csv
  - mon_clean.csv

## Correlations (Spearman)
- WQD: available
- Monteria: available

## Outliers / Range Notes
- WQD: 2.33% of 'temperature' values outside [0,40].; WQD: 4.42% of 'pH' values outside [5.0,9.5].; WQD: temperature has 2.33% robust-z outliers (>3.5).; WQD: pH has 0.70% robust-z outliers (>3.5).; WQD: turbidity_proxy has 1.26% robust-z outliers (>3.5).; WQD: do has 1.26% robust-z outliers (>3.5).; WQD: ammonia has 4.47% robust-z outliers (>3.5).; Monteria: temperature has 0.05% robust-z outliers (>3.5).; Monteria: pH has 0.05% robust-z outliers (>3.5).; Monteria: turbidity has 0.05% robust-z outliers (>3.5).; Monteria: do has 0.05% robust-z outliers (>3.5).

## Next Steps
1) Finalize **cleaning rules** (clip/winsorize, NA handling) per variable using this EDA.
2) Lock **feature space** for Virtual Sensors: inputs = [temperature, pH, turbidity_proxy].
3) Proceed to **01_soft_sensors_DO_NH3.ipynb** to train Virtual DO & NH3 with robust CV.