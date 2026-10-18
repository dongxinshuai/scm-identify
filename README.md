# Environment with py3.8
conda create -n py38 python==3.8\
conda activate py38\
pip install -r py38_requirements.txt

# Environment with py3.12
conda create -n py312 python==3.12\
conda activate py312\
pip install -r py312_requirements.txt

# Structure Identification 
python run_rlcd.py --s multitasking --sample 1 --stage1_method all --n -1 --alpha 0.05 

# Parameter Identification 

(may need multiple run to get ideal results due to random initialization)

python run_params_identify.py --s multitasking --n -1 --method trek --dot_path ./multitasking_results/alpha0.05_rtscale1_N-1.dot

# Results
in ./multitasking_results

# Structure Identification with Nonstationary Data
The input is a list of dataframes

The evacuation scenario reads the four `Pilot4_message_*_inzone_*.csv` files
from `Data/Pilot4_493_four_groups/` relative to this repository.

python run_rlcd.py --s evacuation_cond_data --sample 1 --stage1_method fges --stage1_ges_sparsity 0.2 --n -1 --alpha 0.01 --use_lcdnod 1

Run the focused regression tests with `python -m unittest discover -s tests -v`.
For a full evacuation-data comparison across independent processes and hash seeds,
run `python tests/check_evacuation_reproducibility.py` (results in `.diagnostics/reproducibility/`).



# Reference
"A Versatile Causal Discovery Framework to Allow Causally-Related Hidden Variables." ICLR 2024.

"On the Parameter Identifiability of Partially Observed Linear Causal Models." NeurIPS 2024.

"Identifying Partially Observed Causal Models from Heterogeneous/Nonstationary Data". ICML 2026.
