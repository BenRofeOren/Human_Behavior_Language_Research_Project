import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

df = pd.read_csv('csv_files/segments_features_and_all_features_with_participants_logs_with_difficulty.csv')

df = df.dropna()

""" List of all columns in the data:
       'Unnamed: 0', 'SUBJ', 'SESS', 'TSKN', 'BLKN', 'ACCR', 'TASK', 'TRLN',
       'WORD', 'File_name', 'Reaction_time(s)', 'Total_duration(s)',
       'duration', 'meanIntesnity', 'stddevIntensity', 'meanPitch',
       'stddevPitch', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter',
       'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
       'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer',
       'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean', 'f3_mean',
       'f4_mean', 'f1_median', 'f2_median', 'f3_median', 'f4_median',
       'JitterPCA', 'ShimmerPCA', 'fitch_vtl'"""

df['NO_SEPARATION'] = 1

df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

categorical_features = ['TASK', 'WORD']

# for feature in categorical_features:
#     df[feature] = df[feature].astype('category')
#     df[feature] = df[feature].cat.codes

features_list = ['Reaction_time(s)',
                 'Total_duration(s)', 'duration', 'meanIntesnity', 'stddevIntensity',
                 'meanPitch', 'stddevPitch', 'meanF0Hz', 'stdevF0Hz', 'HNR',
                 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter',
                 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer',
                 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean',
                 'f3_mean', 'f4_mean', 'f1_median', 'f2_median', 'f3_median',
                 'f4_median', 'JitterPCA', 'ShimmerPCA', 'fitch_vtl', 'type',
                 's_first_intensity_mean', 's_first_intensity_std', 's_first_pitch_mean',
                 's_first_pitch_std', 's_first_duration', 's_last_intensity_mean',
                 's_last_intensity_std', 's_last_pitch_mean', 's_last_pitch_std',
                 's_last_duration']

print(df.columns)
split_column_list = ['SUBJ', 'WORD', 'NO_SEPARATION', 'difficulty']

for feature in features_list:
    path_for_plots = 'plots/' + feature
    os.makedirs(path_for_plots)
    for split_column1 in split_column_list:
        path_for_plots_split_column1 = path_for_plots + '/' + split_column1
        os.makedirs(path_for_plots_split_column1)

        # Use groupby to split the DataFrame based on the values in the chosen column
        grouped_df = df.groupby(split_column1)
        print(grouped_df)

        # Iterate through the groups and print them (or perform any desired operation)
        for group_name, group_df in grouped_df:
            # Choose the column to split on
            split_column = 'SESS'

            values_for_list = []
            indecies_for_list = sorted(group_df[split_column].unique())

            print(indecies_for_list)

            # Use groupby to split the DataFrame based on the values in the chosen column
            grouped_df_2 = group_df.groupby(split_column)

            for group_name2, group_df2 in grouped_df_2:
                #print(f"Group {group_name2}:\n{group_df2}\n")
                values_for_list.append(group_df2[feature].mean())
                #print(group_df2[feature].mean())

            # plotting the points
            plt.plot(indecies_for_list, values_for_list)

            # naming the x-axis
            plt.xlabel(split_column)
            # naming the y-axis
            plt.ylabel('average ' + feature)

            # giving a title to my graph
            plt.title(feature + " across " + split_column)
            #plt.grid(True)
            plt.savefig(f'{path_for_plots_split_column1}/{group_name}_{feature}_across_{split_column}.png')
            plt.close()

            #print(f"Group {group_name}:\n{group_df}\n")
