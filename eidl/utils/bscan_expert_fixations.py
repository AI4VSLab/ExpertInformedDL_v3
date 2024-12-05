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


def clean_data_old_format(df):
    filtered_df = df[(df['Sensor'] == 'Eye Tracker') & (df['Eye movement type'] == 'Fixation') & (df['Fixation point X [DACS px]'].notna()) & (df['Fixation point Y [DACS px]'].notna()) & (df['Presented Media name'].str.endswith('.png')) & df['Presented Media name'].str.startswith(('w', 'n'))]
    filtered_df.rename(columns={'Presented Media name': 'image_name'}, inplace=True)
    filtered_df = filtered_df[['Participant name', 'image_name', 'Presented Media width [px]', 'Presented Media height [px]', 'Fixation point X [DACS px]', 'Fixation point Y [DACS px]']]
    return filtered_df


def clean_data_new_format(df, cleaned_response_path):
    df = df[(df['Sensor'] == 'Eye Tracker') & (df['Eye movement type'] == 'Fixation') & (df['Fixation point X [DACS px]'].notna()) & (df['Fixation point Y [DACS px]'].notna())]
    # Calculate the differences in 'Computer timestamp [ms]'
    df['timestamp_diff'] = df['Recording timestamp [ms]'].diff()

    # Initialize columns for fixation_start and fixation_end
    df['fixation_start'] = pd.NaT
    df['fixation_end'] = pd.NaT
    df.reset_index(drop=True, inplace=True)

    df.loc[0, 'fixation_start'] = df.loc[0, 'Recording start time']

    # Calculate fixation_end for each row and fixation_start for the next row
    for i in range(1, len(df)-1):
        # Calculate fixation_end for the previous row
        df.loc[i - 1, 'fixation_end'] = df.loc[i - 1, 'fixation_start'] + pd.to_timedelta(df.loc[i, 'timestamp_diff'], unit='ms')
        # Calculate fixation_start for the current row
        if i < len(df):  # Skip the last row for fixation_start
            df.loc[i, 'fixation_start'] = df.loc[i - 1, 'fixation_end']

    exp_date = df['Recording date'].unique()[0]
    exp_date = pd.to_datetime(exp_date).date()
    df['fixation_start'] = df['fixation_start'].apply(
        lambda t: pd.Timestamp(f"{exp_date} {t}") if pd.notna(t) else pd.NaT
    )
    df['fixation_end'] = df['fixation_end'].apply(
        lambda t: pd.Timestamp(f"{exp_date} {t}") if pd.notna(t) else pd.NaT
    )

    fixation_times=df

    fixation_times['fixation_start'] = pd.to_datetime(fixation_times['fixation_start']).dt.tz_localize(None)
    fixation_times['fixation_end'] = pd.to_datetime(fixation_times['fixation_end']).dt.tz_localize(None)

    # # Image timestamps
    image_times = pd.read_csv(cleaned_response_path, sep=',')
    image_times['image_name'] = image_times['patient_id'].astype(str) + '_' + image_times['image_number'].astype(str) + '.png'

    image_times['page_load_time'] = pd.to_datetime(image_times['page_load_time'], format='%Y-%m-%d_%H-%M-%S.%f')
    image_times['next_button_click_time'] = pd.to_datetime(image_times['next_button_click_time'], format='%Y-%m-%d_%H-%M-%S.%f')

    image_times['page_load_time'] = image_times['page_load_time'] - pd.Timedelta(minutes=3) # Correction was required to match the cleaned response data
    image_times['next_button_click_time'] = image_times['next_button_click_time'] - pd.Timedelta(minutes=3)
    image_times['image_end_time'] = image_times['next_button_click_time']

    # Initialize the 'image_name' column in fixation_times
    fixation_times['image_name'] = None

    # Assign image_name to each fixation point based on the condition
    for i, fixation in fixation_times.iterrows():
        matching_image = image_times[(fixation['fixation_start'] >= image_times['page_load_time']) &
                                    (fixation['fixation_start'] < image_times['image_end_time'])]
        if not matching_image.empty:
            fixation_times.at[i, 'image_name'] = matching_image['image_name'].values[0]

    fixation_times = fixation_times[(fixation_times['Fixation point X [DACS px]'].notna()) & (fixation_times['Fixation point Y [DACS px]'].notna()) & (fixation_times['image_name'].str.endswith('.png')) & fixation_times['image_name'].str.startswith(('w', 'n'))]
    fixation_times = fixation_times[['Participant name', 'image_name', 'Presented Media width [px]', 'Presented Media height [px]', 'Fixation point X [DACS px]', 'Fixation point Y [DACS px]']]
    return fixation_times


def load_all_fixations(root_drive_path_gaze, root_drive_path_cleaned):
    df2 = pd.read_csv(root_drive_path_gaze + 'EEM_2.tsv', sep="\t")
    df3 = pd.read_csv(root_drive_path_gaze + 'EEM_3.tsv', sep="\t")
    df4 = pd.read_csv(root_drive_path_gaze + 'EEM_4.tsv', sep="\t")
    df6 = pd.read_csv(root_drive_path_gaze + 'EEM_6.tsv', sep="\t")
    df7 = pd.read_csv(root_drive_path_gaze + 'EEM_7_2024-09-06.tsv', sep="\t")
    df8_1 = pd.read_csv(root_drive_path_gaze + 'EEM_8_control1_dr_horowitz_experiment_set.tsv', sep="\t")
    df8_2 = pd.read_csv(root_drive_path_gaze + 'EEM_8_control2_dr_horowitz_control.tsv', sep="\t")

    dfs = [df2, df3, df4, df6, df7, df8_1, df8_2]

    old_format_df = dfs[:2]
    old_format_df_filtered = []
    for df in old_format_df:
        old_format_df_filtered.append(clean_data_old_format(df))

    new_format_df = dfs[3:]
    new_format_df_filtered = []
    clean_responses = ['EEM_6_converted.csv', 'EEM_7_converted.csv', 'EEM_8_control1_converted.csv', 'EEM_8_control2_converted.csv']

    for i in range(len(new_format_df)):
        new_format_df_filtered.append(clean_data_new_format(new_format_df[i], root_drive_path_cleaned + clean_responses[i]))

    all_df = old_format_df_filtered + new_format_df_filtered
    final_df = pd.concat(all_df, ignore_index=True)
    final_df['Presented Media width [px]'] = final_df['Presented Media width [px]'].fillna(1920.0)
    final_df['Presented Media height [px]'] = final_df['Presented Media height [px]'].fillna(1080.0)

    final_df.loc[final_df['Presented Media width [px]'] == 1621, 'Fixation point X [DACS px]'] *= (1920 / 1621)
    final_df.loc[final_df['Presented Media width [px]'] == 1621, 'Fixation point X [DACS px]'] = final_df.loc[
        final_df['Presented Media width [px]'] == 1621, 'Fixation point X [DACS px]'
    ].round()
    final_df.loc[final_df['Presented Media width [px]'] == 1621, 'Presented Media width [px]'] = 1920

    # final_df.to_csv('all_data.csv', index=False)

    return final_df

