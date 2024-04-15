# Need to run these in the terminal
# pip install noisereduce
# pip install torch
# pip install openyxl
import math
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os

import noisereduce as nr

import pandas as pd

import re  # for regular expressions
from unidecode import unidecode  # for removing special characters like é

"""# Configurations:"""

# Frame and Hop sizes for computing rms and slicing
FRAME_SIZE = 1024
HOP_LENGTH = 256

# Threshold to cross for rms (in percentage if rms is normalized and in dB otherwise)
THRESHOLD = 0.015
THRESHOLD_END = 0.015

# Maximal amount of frames we allow to go without passing the threshold
MAX_PAUSE_FRAMES = 20

# Whether to normalize rms while slicing
NORMALIZE_RMS = True

# Path of the directory with the example audio files
source_directory = "exampleNewData"

# Specify the path for the new audio file
destination_directory = 'sliced_exampleFiles'


def slice_noise_reduced_audio_files_byt(audio_file, audio_rate, files_name, word,
                                        threshold=THRESHOLD, threshold_end=THRESHOLD_END,
                                        normalize_rms=NORMALIZE_RMS):

    reduced_noise = nr.reduce_noise(y=audio_file, sr=audio_rate, n_std_thresh_stationary=1.5, stationary=True)
    audio_file2 = reduced_noise
    reduced_noise = nr.reduce_noise(y=audio_file2, sr=audio_rate, n_std_thresh_stationary=1.5, stationary=True)

    rms_sound = librosa.feature.rms(y=reduced_noise, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    if np.max(rms_sound) - np.min(rms_sound) == 0:
        return -1, -1, -1

    if normalize_rms:
        rms_sound = (rms_sound - np.min(rms_sound)) / (np.max(rms_sound) - np.min(rms_sound))  # normalising the rms

    frames = range(len(reduced_noise))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    middle = np.argmax(rms_sound)

    start = middle
    flag = True
    while start >= 0 and flag:
        if rms_sound[start] > threshold:
            start -= 1
        else:
            count = 0
            while start - count >= 0 and rms_sound[start-count] < threshold and count < MAX_PAUSE_FRAMES:
                count += 1
            if start - count == -1 or count == MAX_PAUSE_FRAMES:
                flag = False
            else:
                start -= count

    end = middle
    flag = True
    while end < rms_sound.size and flag:
        if rms_sound[end] > threshold_end:
            end += 1
        else:
            count = 0
            while end + count < rms_sound.size and rms_sound[end+count] < threshold_end and count < MAX_PAUSE_FRAMES:
                count += 1
            if end + count == rms_sound.size or count == MAX_PAUSE_FRAMES:
                flag = False
            else:
                end += count

    if word[-1] != 'a' and word[-1] != 'e' and word[-1] != 'i' and word[-1] != 'o' and word[-1] != 'u':
        end += 12
        if end >= rms_sound.size:
            end = rms_sound.size - 1

    sound_s = audio_file[start * HOP_LENGTH: end * HOP_LENGTH + FRAME_SIZE]
    #print("Sliced " + files_name + " at times:", t[start], t[end], "total size:", sound_s.size, " frames sliced: ", start, end)
    total_duration = t[end] - t[start]
    return sound_s, t[start], total_duration


def load_audio_files(samples_folder):
    folder_content = os.listdir(samples_folder)
    loaded_audio = []
    loaded_sr = []
    for file in folder_content:
        print("Loading " + file)
        loaded_audio.append(librosa.load(samples_folder + "/" + file)[0])
        loaded_sr.append(librosa.load(samples_folder + "/" + file)[1])
    return loaded_audio, loaded_sr


def process_wav_file(input_file, output_directory, relative_path, robot_times):
    # The function slices the file and saves the result in a similar path to the original
    # The function returns features that cannot be extracted after the slicing is done, like reaction time

    if not input_file.endswith(".wav"):
        return

    audio_data, sample_rate = librosa.load(input_file)
    file_name = input_file

    pattern = r'\((.*?)\)'  # This regular expression captures content inside parentheses
    match = re.search(pattern, file_name)

    word = match.group(1)

    # Modify content as needed
    sliced_audio_file, onset, total_duration = slice_noise_reduced_audio_files_byt(audio_file=audio_data, word=word,
                                                                                   audio_rate=sample_rate,
                                                                                   files_name=file_name)

    real_onset = onset - robot_times[word]


    to_include = not (real_onset > 2 or real_onset < -0.5 or total_duration < 0.3 or total_duration > 3)

    output_file = os.path.join(output_directory, relative_path)

    features = [repr(output_file), real_onset, total_duration]

    # Create the output file with the same relative path in the specified directory
    if to_include:
        sf.write(output_file, sliced_audio_file, int(sample_rate))
    else:
        print("not included: " + output_file)

    #print(word, "onset: ", onset, "real onset: ", real_onset)

    #print("saved:", file_name)

    return features, to_include


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def copy_directory_with_wav_processing(source_directory, destination_directory, robot_times):
    # Copy the directory tree
    shutil.copytree(source_directory, destination_directory, ignore=ig_f)  # copying only the directory structure

    count = 0

    slice_features = pd.DataFrame(columns=['File_name', 'Reaction_time(s)', 'Total_duration(s)'])

    # Walk through the copied directory and process each WAV file
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith('.wav'):
                if count % 100 == 0:
                    print("copied ", count, " files")
                if count == 500:
                    break
                #print(file)
                file_path = os.path.join(root, file)
                #print(file_path)
                # Get the relative path within the directory structure
                relative_path = os.path.relpath(file_path, source_directory)
                #print(relative_path)

                file_features, to_include = process_wav_file(file_path, destination_directory,
                                                             relative_path, robot_times)
                #print(file_features)
                if to_include:
                    slice_features.loc[len(slice_features.index)] = file_features

                #print()
                count += 1
    return slice_features


# get the duration of each recording of the computer
df = pd.read_excel('Excels/Stimuli Duration.xlsx')
robot_times = df.set_index('Word')['Modified Duration'].to_dict()
robot_times = {unidecode(key).lower(): value for key, value in robot_times.items()}
#print(robot_times)


sliced_features = copy_directory_with_wav_processing(source_directory, destination_directory, robot_times)
sliced_features.to_csv("csv_files/Sliced_features.csv")
print(sliced_features.to_string())










