import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

df = pd.read_csv('csv_files/segments_features_and_all_features_with_participants_logs_with_difficulty.csv')

df = df.dropna()

df['NO_SEPARATION'] = 1

df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

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
split_label_column_list = ['SESS', 'ACCR', 'COND']


for feature in features_list:
    path_for_plots = 'plots_histogram/' + feature
    os.makedirs(path_for_plots)
    for split_column1 in split_column_list:
        path_for_plots_split_column1 = path_for_plots + '/' + split_column1
        os.makedirs(path_for_plots_split_column1)

        # Use groupby to split the DataFrame based on the values in the chosen column
        grouped_df = df.groupby(split_column1)
        print(grouped_df)

        # Choose the column to split on
        for split_label_column in split_label_column_list:

            path_for_plots_split_label_column = path_for_plots_split_column1 + '/' + split_label_column
            os.makedirs(path_for_plots_split_label_column)

            # Iterate through the groups and print them (or perform any desired operation)
            for group_name, group_df in grouped_df:

                values_for_list = []
                indecies_for_list = sorted(group_df[split_label_column].unique())

                print(indecies_for_list)

                # Use groupby to split the DataFrame based on the values in the chosen column
                grouped_df_2 = group_df.groupby(split_label_column)

                for group_name2, group_df2 in grouped_df_2:
                    # print(f"Group {group_name2}:\n{group_df2}\n")
                    # Plotting a basic histogram
                    plt.hist(group_df2[feature], alpha=0.5, label=str(group_name2))

                plt.legend()

                # naming the x-axis
                plt.xlabel(feature)
                # naming the y-axis
                plt.ylabel("amount")

                # giving a title to my graph
                plt.title(feature + " histogram for each " + split_label_column + " for " + split_column1)
                #plt.grid(True)
                plt.savefig(f'{path_for_plots_split_label_column}/{split_column1}_{group_name}_{feature}_across_'
                            f'{split_label_column}.png')
                plt.close()

                #print(f"Group {group_name}:\n{group_df}\n")
