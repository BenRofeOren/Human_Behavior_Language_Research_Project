# This code reads the csv files from segmentation_separation_csv directory and extracts the information
# and calculates the features from the segments

import pandas as pd
import pickle
import os
import librosa
import re

import parselmouth
from parselmouth.praat import call

import numpy as np

import statistics


# def last_letter_is_vowel(x_sampa_string):
#     # X-SAMPA representation of vowels
#     x_sampa_vowels = {'i', 'I', 'I\\', 'e', 'E', 'a', 'A', 'O', 'o', 'u', 'U', 'M', 'Q', 'U\\', 'V', 'Y'
#                         '@\\', '{', '}', '1', '2', '3', '3\\', '6', '7', '8', '9', '&', '/'}
#
#     # Convert the input string to lowercase
#     x_sampa_string = x_sampa_string.lower()
#
#     # Check if the last character of the string is a vowel
#     if x_sampa_string[-1] in x_sampa_vowels:
#         return 1
#     else:
#         return 0


# This function calculates auditory features based on the division to segments of the data
# This function returns whatever it needs to turn it into a row of a pandas dataframe
# (probably a dictionary of features)
def calculate_features(segments_df, file_path_original):
    # Do fancy calculations:
    # For the last 2 segments we extract their duration and the mean and std of their intensity, pitch and formants
    # print("Original File Path:", repr(file_path_original))

    # Loading the matching file
    # Convert backslashes to forward slashes for cross-platform compatibility
    file_path = file_path_original.replace('\\\\', '/')

    file_path = file_path[1:-1]
    # print("File Path:", file_path)

    if not os.path.exists(file_path):
        print("not found: ", file_path)
        return {}

    audio_data, sample_rate = librosa.load(file_path)
    # print(len(audio_data))

    # Remove rows containing the "<p:>" in "MAU"
    segments_df = segments_df[~segments_df["MAU"].str.contains("<p:>")]

    # # Get the duration of the first segment
    # s1_duration = segments_df.iloc[0]["DURATION"]
    #
    # # Get the duration of the last segment
    # s2_duration = segments_df.iloc[-1]["DURATION"]

    # Get the audio of each segment
    #segments_df["Audio"] = audio_data[segments_df["BEGIN"]: segments_df["BEGIN"] + segments_df["DURATION"]]

    # print(audio_data)

    attributes = {}

    # X-SAMPA representation of vowels
    x_sampa_vowels = {'i', 'I', 'I\\', 'e', 'E', 'a', 'A', 'O', 'o', 'u', 'U', 'M', 'Q', 'U\\', 'V', 'Y'
                      '@\\', '{', '}', '1', '2', '3', '3\\', '6', '7', '8', '9', '&', '/', 'u:', 'i:',
                      '@U'}

    pattern = r'\((.*?)\)'  # This regular expression captures content inside parentheses
    match = re.search(pattern, file_path)

    word = match.group(1)

    if segments_df.iloc[-1]["MAU"] in x_sampa_vowels:
        type_of_word = 1
    else:
        type_of_word = 0

    # print(word, segments_df.iloc[-1]["MAU"], type_of_word)

    attributes["type"] = type_of_word

    time_step = 0.
    min_time = 0.
    max_time = 0.
    pitch_floor = 101.
    # pitch_floor = 100.

    pitch_ceiling = 600.
    unit = 'Hertz'

    # f0min, f0max = 75, 300
    f0min, f0max = pitch_floor, 300

    last_index = segments_df.shape[0]

    segments_to_calculate = [0, last_index-1]
    for i in segments_to_calculate:
        si = segments_df.iloc[i]
        si_audio = audio_data[si["BEGIN"]: si["BEGIN"] + si["DURATION"]]

        # Extract intensity (loudness) features
        rms_window = int(len(si_audio) / 10)  # Set window size to 1/10th of the signal length
        intensity = librosa.feature.rms(y=si_audio, frame_length=rms_window)

        # Extract pitch features
        hop_length = 128  # 512  # Adjust as needed
        pitches, magnitudes = librosa.piptrack(y=si_audio, sr=sample_rate,
                                               hop_length=hop_length, n_fft=min(2048, len(si_audio)))

        index = "first"
        if i == segments_df.shape[0]-1:
            index = "last"

        attributes['s_'+str(index)+"_intensity_mean"] = np.mean(intensity)
        attributes['s_'+str(index)+"_intensity_std"] = np.std(intensity)
        attributes['s_'+str(index)+"_pitch_mean"] = np.mean(pitches)
        attributes['s_'+str(index)+"_pitch_std"] = np.std(pitches)
        attributes['s_'+str(index)+"_duration"] = si["DURATION"]

        # sound = parselmouth.Sound(si_audio)  # read the sound
        #
        # # # Extracting the mean and std of the intensity, pitch and formants
        # # intensity = call(sound, 'To Intensity', pitch_floor, time_step, 'yes')
        # #
        # # attributes['s'+str(i)+'mean_intensity'] = call(intensity, 'Get mean',
        # #                                                min_time, max_time)
        # #
        # # attributes['s'+str(i)+'stddev_intensity'] = call(intensity, 'Get standard deviation',
        # #                                                  min_time, max_time)
        #
        # pitch = call(sound, 'To Pitch', time_step, pitch_floor, pitch_ceiling)
        #
        # attributes['s'+str(i)+'mean_pitch'] = call(pitch, 'Get mean',
        #                                            min_time, max_time,
        #                                            unit)
        #
        # attributes['s'+str(i)+'stddev_pitch'] = call(pitch, 'Get standard deviation',
        #                                              min_time, max_time,
        #                                              unit)
        #
        # pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        #
        # formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        # numPoints = call(pointProcess, "Get number of points")
        #
        # f1_list = []
        # f2_list = []
        # f3_list = []
        # f4_list = []
        #
        # # Measure formants only at glottal pulses
        # for point in range(0, numPoints):
        #     point += 1
        #     t = call(pointProcess, "Get time from index", point)
        #     f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        #     f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        #     f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        #     f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        #     f1_list.append(f1)
        #     f2_list.append(f2)
        #     f3_list.append(f3)
        #     f4_list.append(f4)
        #
        # f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
        # f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
        # f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
        # f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
        #
        # # calculate mean formants across pulses
        # if not f1_list:
        #     attributes['s' + str(i) + 'f1_mean'] = None
        #     attributes['s' + str(i) + 'f2_mean'] = None
        #     attributes['s' + str(i) + 'f3_mean'] = None
        #     attributes['s' + str(i) + 'f4_mean'] = None
        # else:
        #     attributes['s'+str(i)+'f1_mean'] = statistics.mean(f1_list)
        #     attributes['s'+str(i)+'f2_mean'] = statistics.mean(f2_list)
        #     attributes['s'+str(i)+'f3_mean'] = statistics.mean(f3_list)
        #     attributes['s'+str(i)+'f4_mean'] = statistics.mean(f4_list)
        #
        # # print(si_audio)

    return attributes


csv_directory = "segmentation_separation_csv_exampleNewData"

# Load the dictionary that translate csv file to actual name
with open('name_to_num.pkl', 'rb') as handle:
    name_to_num = pickle.load(handle)

# Load the dictionary that translate csv file to actual name
with open('num_to_name.pkl', 'rb') as handle:
    num_to_name = pickle.load(handle)

#print(num_to_name)

segments_features_csv = pd.read_csv("csv_files/segments_features.csv")

all_extracted_features = []

count = 0
amount = len(os.listdir(csv_directory))
for filename in os.listdir(csv_directory):
    f = os.path.join(csv_directory, filename)

    count += 1

    if count == 1:
        continue

    if count % 200 == 0:
        print("processed ", count, " files out of ", amount)

    df = pd.read_csv(f, sep=";")
    #print(df)
    df = df[["BEGIN", "DURATION",  "MAU", "ORT"]]
    # print(df.to_string())
    # print(filename[:-4])
    file_name = num_to_name[int(filename[:-4])]
    # print(file_name)

    #if segments_features_csv["File_name"].isin(file_name).any():
    #    continue

    extracted_features1 = calculate_features(df, file_name)

    if extracted_features1:  # is the dictionary not empty
        extracted_features = {"File_name": file_name, **extracted_features1}
        all_extracted_features.append(extracted_features)
    # print("\n\n")

all_extracted_features_csv = pd.DataFrame(all_extracted_features)
print(all_extracted_features_csv)
all_extracted_features_csv.to_csv("csv_files/segments_features.csv", index=False)


