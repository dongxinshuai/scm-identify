# Environment
conda create -n myenv python==3.8\
conda activate myenv\
pip install -r requirements.txt

# Structure Identification 
python run_rlcd.py --s multitasking --sample 1 --stage1_method all --n -1 --alpha 0.05 

# Parameter Identification 
python run_params_identify.py --s multitasking --n -1 --method trek --dot_path ./multitasking_results/alpha0.05_rtscale1_N-1.dot

# Results
in ./multitasking_results

