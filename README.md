conda create -n myenv python==3.8
conda activate myenv
pip install -r requirements.txt

# Structure Identification
python run_rlcd.py --s multitasking --n -1 --stage1_method all --alpha 0.05 --sample 1

# Parameter Identification
python run_params_identify.py --s multitasking --n -1 --method trek --dot_path multitasking_results/alpha0.05_rtscale1_N-1.dot

# result in multitasking_results