import os

# # pairwise curation of CUE0_lte2400 data (pairwise in so far as we only keep trials that
# # were clean in both PFC and HPC)
# os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400/Vehicle_coke_PFC.npy data_aggregated/CUE0_lte2400/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400/Vehicle_saline_PFC.npy data_aggregated/CUE0_lte2400/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400/ChABC_coke_PFC.npy data_aggregated/CUE0_lte2400/ChABC_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400/ChABC_saline_PFC.npy data_aggregated/CUE0_lte2400/ChABC_saline_HPC.npy')
#
# # repeat for CUE1
# os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400/Vehicle_coke_PFC.npy data_aggregated/CUE1_lte2400/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400/Vehicle_saline_PFC.npy data_aggregated/CUE1_lte2400/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400/ChABC_coke_PFC.npy data_aggregated/CUE1_lte2400/ChABC_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400/ChABC_saline_PFC.npy data_aggregated/CUE1_lte2400/ChABC_saline_HPC.npy')
#
# # T1 (coke only this day)
# os.system('python3 curate_trials.py data_aggregated/T1/Vehicle_coke_PFC.npy data_aggregated/T1/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T1/ChABC_coke_PFC.npy data_aggregated/T1/ChABC_coke_HPC.npy')
#
# # T2 (saline only this day)
# os.system('python3 curate_trials.py data_aggregated/T2/Vehicle_saline_PFC.npy data_aggregated/T2/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T2/ChABC_saline_PFC.npy data_aggregated/T2/ChABC_saline_HPC.npy')
#
# # T7 (coke only this day)
# os.system('python3 curate_trials.py data_aggregated/T7/Vehicle_coke_PFC.npy data_aggregated/T7/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T7/ChABC_coke_PFC.npy data_aggregated/T7/ChABC_coke_HPC.npy')
#
# # T8 (saline only this day)
# os.system('python3 curate_trials.py data_aggregated/T8/Vehicle_saline_PFC.npy data_aggregated/T8/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T8/ChABC_saline_PFC.npy data_aggregated/T8/ChABC_saline_HPC.npy')
#
# ## long epochs (9s)
# # T1 (coke only this day)
# os.system('python3 curate_trials.py data_aggregated/T1_long/Vehicle_coke_PFC.npy data_aggregated/T1_long/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T1_long/ChABC_coke_PFC.npy data_aggregated/T1_long/ChABC_coke_HPC.npy')
#
# # T2 (saline only this day)
# os.system('python3 curate_trials.py data_aggregated/T2_long/Vehicle_saline_PFC.npy data_aggregated/T2_long/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T2_long/ChABC_saline_PFC.npy data_aggregated/T2_long/ChABC_saline_HPC.npy')
#
# # T7 (coke only this day)
# os.system('python3 curate_trials.py data_aggregated/T7_long/Vehicle_coke_PFC.npy data_aggregated/T7_long/Vehicle_coke_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T7_long/ChABC_coke_PFC.npy data_aggregated/T7_long/ChABC_coke_HPC.npy')
#
# # T8 (saline only this day)
# os.system('python3 curate_trials.py data_aggregated/T8_long/Vehicle_saline_PFC.npy data_aggregated/T8_long/Vehicle_saline_HPC.npy')
# os.system('python3 curate_trials.py data_aggregated/T8_long/ChABC_saline_PFC.npy data_aggregated/T8_long/ChABC_saline_HPC.npy')

# Long CUE days
# CUE0
os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400_long/Vehicle_coke_PFC.npy data_aggregated/CUE0_lte2400_long/Vehicle_coke_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400_long/Vehicle_saline_PFC.npy data_aggregated/CUE0_lte2400_long/Vehicle_saline_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400_long/ChABC_coke_PFC.npy data_aggregated/CUE0_lte2400_long/ChABC_coke_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE0_lte2400_long/ChABC_saline_PFC.npy data_aggregated/CUE0_lte2400_long/ChABC_saline_HPC.npy')

# repeat for CUE1
os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400_long/Vehicle_coke_PFC.npy data_aggregated/CUE1_lte2400_long/Vehicle_coke_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400_long/Vehicle_saline_PFC.npy data_aggregated/CUE1_lte2400_long/Vehicle_saline_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400_long/ChABC_coke_PFC.npy data_aggregated/CUE1_lte2400_long/ChABC_coke_HPC.npy')
os.system('python3 curate_trials.py data_aggregated/CUE1_lte2400_long/ChABC_saline_PFC.npy data_aggregated/CUE1_lte2400_long/ChABC_saline_HPC.npy')
