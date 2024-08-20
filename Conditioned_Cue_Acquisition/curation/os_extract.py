import os
import sys
import argparse

parser = argparse.ArgumentParser(description='script to execute extract_data.py. Takes day as input denoting which recording sessions to extract')
parser.add_argument('day', metavar='<day>', help='str denoting recording session to extract')
args = parser.parse_args()

def main():
    if args.day == 'CUE0':
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys5_rat1_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys5_rat2_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys6_rat1_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys7_rat2_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys9_rat1_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys9_rat2_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys10_rat1_CUE0/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE0/Ephys11_rat1_CUE0/')

    if args.day == 'CUE1':
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys5_rat1_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys5_rat2_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys6_rat1_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys7_rat2_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys9_rat1_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys9_rat2_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys10_rat1_CUE1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/CUE1/Ephys11_rat1_CUE1/')

    if args.day == 'T1':
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys5_rat1_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys5_rat2_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys6_rat1_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys7_rat2_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys9_rat1_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys9_rat2_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys10_rat1_T1/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T1/Ephys11_rat1_T1/')

    if args.day == 'T2':
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys5_rat1_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys5_rat2_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys6_rat1_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys7_rat2_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys9_rat1_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys9_rat2_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys10_rat1_T2/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T2/Ephys11_rat1_T2/')

    if args.day == 'T7':
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys5_rat1_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys5_rat2_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys6_rat1_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys7_rat2_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys9_rat1_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys9_rat2_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys10_rat1_T7/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T7/Ephys11_rat1_T7/')

    if args.day == 'T8':
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys5_rat1_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys5_rat2_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys6_rat1_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys7_rat2_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys9_rat1_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys9_rat2_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys10_rat1_T8/')
        os.system('python3 extract_data.py /Volumes/T7/CCA_Raw_Data/T8/Ephys11_rat1_T8/')

    print('\nextraction complete!')
    sys.exit()

if __name__ == '__main__':
    main()
