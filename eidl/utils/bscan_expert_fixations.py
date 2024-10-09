#Cleaning for Old Way, this way is not used to clean new doctors was used early on for when the experiment was different
# Dr. Diaconita, Dr. Chen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_all_karen_fixations(file_path):
    """Load all fixations from the given file path.

    Args:
        file_path (str): The file path to the data file.

    Returns:


    """
    df = pd.read_csv(file_path, sep='\t')
    # Filter data initially for Eye Tracker and Fixations that are not NA
    filtered_df = df[(df['Sensor'] == 'Eye Tracker') & (df['Eye movement type'] == 'Fixation') & (df['Fixation point X [DACS px]'].notna()) & (df['Fixation point Y [DACS px]'].notna()) & ((df['Participant name'] == 'dr. diaconita') | (df['Participant name'] == 'royce')) & (df['Presented Media name'].str.endswith('.png'))]
    filtered_df.rename(columns={'Presented Media name': 'image_name'}, inplace=True)
    filtered_df = filtered_df[['Participant name', 'image_name', 'Presented Media width [px]', 'Presented Media height [px]', 'Fixation point X [DACS px]', 'Fixation point Y [DACS px]']]


    #Cleaning for New Way - the time of next image is not signified, so we need to match it with when the image changes
    #Dr B. and all further doctors

    #Fixation timestamps
    df = pd.read_csv('/data/leo/data/BScan/ExpertEyetracking/dr_b_control_set_1.tsv', sep='\t')
    df = df[(df['Sensor'] == 'Eye Tracker') & (df['Eye movement type'] == 'Fixation') & (df['Fixation point X [DACS px]'].notna()) & (df['Fixation point Y [DACS px]'].notna())]

    # Calculate the differences in 'Computer timestamp [ms]'
    df['timestamp_diff'] = df['Recording timestamp [ms]'].diff()

    # Initialize columns for fixation_start and fixation_end
    df['fixation_start'] = pd.NaT
    df['fixation_end'] = pd.NaT
    df.reset_index(drop=True, inplace=True)

    df.loc[0, 'fixation_start'] = df.loc[0, 'Recording start time UTC']

    # Calculate fixation_end for each row and fixation_start for the next row
    for i in range(1, len(df)-1):
        # Calculate fixation_end for the previous row
        df.loc[i - 1, 'fixation_end'] = df.loc[i - 1, 'fixation_start'] + pd.to_timedelta(df.loc[i, 'timestamp_diff'], unit='ms')
        # Calculate fixation_start for the current row
        if i < len(df):  # Skip the last row for fixation_start
            df.loc[i, 'fixation_start'] = df.loc[i - 1, 'fixation_end']

    fixation_times = df

    #Image timestamps
    image_times = pd.read_csv("/data/leo/data/BScan/ExpertEyetracking/dr b control set 1_times.csv", sep=',')
    image_times['image_name'] = image_times['patient_id'].astype(str) + '_' + image_times['image_number'].astype(str) + '.png'

    image_times['page_load_time'] = pd.to_datetime(image_times['page_load_time'], utc=True)
    image_times['next_button_click_time'] = pd.to_datetime(image_times['next_button_click_time'], utc=True)

    image_times['page_load_time'] = image_times['page_load_time'] - pd.Timedelta(minutes=3)
    image_times['next_button_click_time'] = image_times['next_button_click_time'] - pd.Timedelta(minutes=3)

    #Pairing fixation times with image_names

    fixation_times['fixation_start'] = pd.to_datetime(fixation_times['fixation_start'], utc=True)
    fixation_times['fixation_end'] = pd.to_datetime(fixation_times['fixation_end'], utc=True)

    image_times['image_end_time'] = image_times['next_button_click_time']

    # Initialize the 'image_name' column in fixation_times
    fixation_times['image_name'] = None

    # Assign image_name to each fixation point based on the condition
    for i, fixation in fixation_times.iterrows():
        matching_image = image_times[(fixation['fixation_start'] >= image_times['page_load_time']) &
                                     (fixation['fixation_start'] < image_times['image_end_time'])]
        if not matching_image.empty:
            fixation_times.at[i, 'image_name'] = matching_image['image_name'].values[0]

    fixation_times = fixation_times[['Participant name', 'image_name', 'Presented Media width [px]', 'Presented Media height [px]', 'Fixation point X [DACS px]', 'Fixation point Y [DACS px]']]
    df_combined = pd.concat([filtered_df, fixation_times], ignore_index=True)
    return df_combined