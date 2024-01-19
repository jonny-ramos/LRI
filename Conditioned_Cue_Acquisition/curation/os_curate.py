import os

# pairwise curation of CUE0 data (pairwise in so far as we only keep trials that
# were clean in both PFC and HPC)
# os.system('python3 curate_trials.py data_aggregated/CUE0/Vehicle_coke_PFC.npy data_aggregated/CUE0/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE0/Vehicle_saline_PFC.npy data_aggregated/CUE0/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE0/ChABC_coke_PFC.npy data_aggregated/CUE0/ChABC_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE0/ChABC_saline_PFC.npy data_aggregated/CUE0/ChABC_saline_HPC.npy')
#
# # repeat for CUE1
os.system('python3 curate_trials.py data_aggregated/CUE1/Vehicle_coke_PFC.npy data_aggregated/CUE1/Vehicle_coke_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE1/Vehicle_saline_PFC.npy data_aggregated/CUE1/Vehicle_saline_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE1/ChABC_coke_PFC.npy data_aggregated/CUE1/ChABC_coke_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE1/ChABC_saline_PFC.npy data_aggregated/CUE1/ChABC_saline_HPC.npy')
