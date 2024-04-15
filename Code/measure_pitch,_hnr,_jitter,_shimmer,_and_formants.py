#pip install praat-parselmouth

import glob
import numpy as np
import pandas as pd
import parselmouth
import statistics


from parselmouth.praat import call
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os


"""## This function measures duration, pitch, HNR, jitter, and shimmer"""
def get_intensity_attributes(sound, time_step=0., min_time=0., max_time=0., pitch_floor=75.,
                             interpolation_method='Parabolic', return_values=False,
                             replacement_for_nan=0.):
    """
    Function to get intensity attributes such as minimum intensity, maximum intensity, mean
    intensity, and standard deviation of intensity.
    NOTE: Notice that we don't need a unit parameter for intensity as intensity is consistently
    reported as dB SPL throughout Praat. dB SPL is simply dB relative to the normative auditory
    threshold for a 1000-Hz sine wave, 2 x 10^(-5) Pascal.
    NOTE: The standard interpolation method is 'Parabolic' because of the usual non-linearity
    (logarithm) in the computation of intensity; sinc interpolation would be too stiff and may
    give unexpected results.
    :param (parselmouth.Sound) sound: sound waveform
    :param (float) time_step: the measurement interval (frame duration), in seconds (default: 0.)
           NOTE: The default 0. value corresponds to a time step of 0.75 / pitch floor
    :param (float) min_time: minimum time value considered for time range (t1, t2) (default: 0.)
    :param (float) max_time: maximum time value considered for time range (t1, t2) (default: 0.)
           NOTE: If max_time <= min_time, the entire time domain is considered
    :param pitch_floor: minimum pitch (default: 75.)
    :param (str) interpolation_method: method of sampling new data points with a discrete set of
           known data points, 'None', 'Parabolic', 'Cubic', or 'Sinc' (default: 'Parabolic')
    :param (bool) return_values: whether to return a continuous list of intensity values
           from all frames or not
    :param (float) replacement_for_nan: a float number that will represent frames with NaN values
    :return: (a dictionary of mentioned attributes, a list of intensity values OR None)
    """
    # Get total duration of the sound
    duration = call(sound, 'Get end time')

    # Create Intensity object
    intensity = call(sound, 'To Intensity', pitch_floor, time_step, 'yes')

    attributes = dict()

    attributes['min_intensity'] = call(intensity, 'Get minimum',
                                       min_time, max_time,
                                       interpolation_method)

    attributes['relative_min_intensity_time'] = call(intensity, 'Get time of minimum',
                                                     min_time, max_time,
                                                     interpolation_method) / duration

    attributes['max_intensity'] = call(intensity, 'Get maximum',
                                       min_time, max_time,
                                       interpolation_method)

    attributes['relative_max_intensity_time'] = call(intensity, 'Get time of maximum',
                                                     min_time, max_time,
                                                     interpolation_method) / duration

    attributes['mean_intensity'] = call(intensity, 'Get mean',
                                        min_time, max_time)

    attributes['stddev_intensity'] = call(intensity, 'Get standard deviation',
                                          min_time, max_time)

    attributes['q1_intensity'] = call(intensity, 'Get quantile',
                                      min_time, max_time,
                                      0.25)

    attributes['median_intensity'] = call(intensity, 'Get quantile',
                                          min_time, max_time,
                                          0.50)

    attributes['q3_intensity'] = call(intensity, 'Get quantile',
                                      min_time, max_time,
                                      0.75)

    intensity_values = None

    if return_values:
        intensity_values = [call(intensity, 'Get value in frame', frame_no)
                            for frame_no in range(len(intensity))]
        # Convert NaN values to floats (default: 0)
        intensity_values = [value if not math.isnan(value) else replacement_for_nan
                            for value in intensity_values]

    return attributes,  intensity_values

def get_pitch_attributes(sound, pitch_type='preferred', time_step=0., min_time=0., max_time=0.,
                         pitch_floor=75., pitch_ceiling=600., unit='Hertz',
                         interpolation_method='Parabolic', return_values=False,
                         replacement_for_nan=0.):
    """
    Function to get pitch attributes such as minimum pitch, maximum pitch, mean pitch, and
    standard deviation of pitch.
    :param (parselmouth.Sound) sound: sound waveform
    :param (str) pitch_type: the type of pitch analysis to be performed; values include 'preferred'
           optimized for speech based on auto-correlation method, and 'cc' for performing acoustic
           periodicity detection based on cross-correlation method
           NOTE: Praat also includes an option for type 'ac', a variation of 'preferred' that
           requires several more parameters. We are not including this for simplification.
    :param (float) time_step: the measurement interval (frame duration), in seconds (default: 0.)
           NOTE: The default 0. value corresponds to a time step of 0.75 / pitch floor
    :param (float) min_time: minimum time value considered for time range (t1, t2) (default: 0.)
    :param (float) max_time: maximum time value considered for time range (t1, t2) (default: 0.)
           NOTE: If max_time <= min_time, the entire time domain is considered
    :param (float) pitch_floor: minimum pitch (default: 75.)
    :param (float) pitch_ceiling: maximum pitch (default: 600.)
    :param (str) unit: units of the result, 'Hertz' or 'Bark' (default: 'Hertz)
    :param (str) interpolation_method: method of sampling new data points with a discrete set of
           known data points, 'None' or 'Parabolic' (default: 'Parabolic')
    :param (bool) return_values: whether to return a continuous list of pitch values from all frames
           or not
    :param (float) replacement_for_nan: a float number that will represent frames with NaN values
    :return: (a dictionary of mentioned attributes, a list of pitch values OR None)
    """
    # Get total duration of the sound
    duration = call(sound, 'Get end time')

    # Create pitch object
    if pitch_type == 'preferred':
        pitch = call(sound, 'To Pitch', time_step, pitch_floor, pitch_ceiling)
    elif pitch_type == 'cc':
        pitch = call(sound, 'To Pitch (cc)', time_step, pitch_floor, pitch_ceiling)
    else:
        raise ValueError('Argument for @pitch_type not recognized!')

    attributes = dict()

    attributes['voiced_fraction'] = call(pitch, 'Count voiced frames') / len(pitch)

    attributes['min_pitch'] = call(pitch, 'Get minimum',
                                   min_time, max_time,
                                   unit,
                                   interpolation_method)

    attributes['relative_min_pitch_time'] = call(pitch, 'Get time of minimum',
                                                 min_time, max_time,
                                                 unit,
                                                 interpolation_method) / duration

    attributes['max_pitch'] = call(pitch, 'Get maximum',
                                   min_time, max_time,
                                   unit,
                                   interpolation_method)

    attributes['relative_max_pitch_time'] = call(pitch, 'Get time of maximum',
                                                 min_time, max_time,
                                                 unit,
                                                 interpolation_method) / duration

    attributes['mean_pitch'] = call(pitch, 'Get mean',
                                    min_time, max_time,
                                    unit)

    attributes['stddev_pitch'] = call(pitch, 'Get standard deviation',
                                      min_time, max_time,
                                      unit)

    attributes['q1_pitch'] = call(pitch, 'Get quantile',
                                  min_time, max_time,
                                  0.25,
                                  unit)

    attributes['median_intensity'] = call(pitch, 'Get quantile',
                                          min_time, max_time,
                                          0.50,
                                          unit)

    attributes['q3_pitch'] = call(pitch, 'Get quantile',
                                  min_time, max_time,
                                  0.75,
                                  unit)

    attributes['mean_absolute_pitch_slope'] = call(pitch, 'Get mean absolute slope', unit)
    attributes['pitch_slope_without_octave_jumps'] = call(pitch, 'Get slope without octave jumps')

    pitch_values = None

    if return_values:
        pitch_values = [call(pitch, 'Get value in frame', frame_no, unit)
                        for frame_no in range(len(pitch))]
        # Convert NaN values to floats (default: 0)
        pitch_values = [value if not math.isnan(value) else replacement_for_nan
                        for value in pitch_values]

    return attributes, pitch_values


# This is the function to measure source acoustics using default male parameters.
def measurePitch(File_name, f0min, f0max, unit):
    sound = parselmouth.Sound(File_name) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    intensity_att, intensity_val = get_intensity_attributes(sound)
    pitch_att, pitch_val = get_pitch_attributes(sound)

    meanIntensity = intensity_att['mean_intensity']
    stddevIntensity = intensity_att['stddev_intensity']

    meanPitch = pitch_att['mean_pitch']
    stddevPitch = pitch_att['stddev_pitch']

    #print(duration, meanIntensity, stddevIntensity, meanPitch, stddevPitch, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer)

    return duration, meanIntensity, stddevIntensity, meanPitch, stddevPitch, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


# This function measures formants using Formant Position formula
def measureFormants(sound, wave_file, f0min,f0max):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

    # calculate mean formants across pulses
    if not f1_list:
        return None, None, None, None, None, None, None, None

    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)

    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)

    return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median

"""## This function runs a 2-factor Principle Components Analysis (PCA) on Jitter and Shimmer"""
def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    df = df.dropna()
    x = df.loc[:, measures].values
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['JitterPCA', 'ShimmerPCA'])
    return principalDf

"""## This block of code runs the above functions on all of the '.wav' files in the /audio folder"""

# create lists to put the results
file_list = []
duration_list = []
meanIntensity_list = [] #new
stddevIntensity_list = [] #new
meanPitch_list = [] #new
stddevPitch_list = [] #new
mean_F0_list = []
sd_F0_list = []
hnr_list = []
localJitter_list = []
localabsoluteJitter_list = []
rapJitter_list = []
ppq5Jitter_list = []
ddpJitter_list = []
localShimmer_list = []
localdbShimmer_list = []
apq3Shimmer_list = []
aqpq5Shimmer_list = []
apq11Shimmer_list = []
ddaShimmer_list = []
f1_mean_list = []
f2_mean_list = []
f3_mean_list = []
f4_mean_list = []
f1_median_list = []
f2_median_list = []
f3_median_list = []
f4_median_list = []


source_directory = "sliced_exampleFiles"
#source_directory = "exampleNewData"

count = 0

# Go through all the wave files in the folder and measure all the acoustics
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
            wave_file = file_path
            sound = parselmouth.Sound(wave_file)
            attributes, intensity_values = get_intensity_attributes(sound)

            # print(attributes, intensity_values)
            # print()
            (duration, meanIntensity, stddevIntensity, meanPitch, stddevPitch, meanF0, stdevF0, hnr, localJitter,
             localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
             localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(
                sound, 75, 300, "Hertz")
            #print("sound:", sound)
            (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median) = measureFormants(
                sound, wave_file, 75, 300)
            file_list.append(repr(wave_file))  # make an ID list
            duration_list.append(duration)  # make duration list
            meanIntensity_list.append(meanIntensity)  # make a mean intensity list
            stddevIntensity_list.append(stddevIntensity)  # new
            meanPitch_list.append(meanPitch)  # new
            stddevPitch_list.append(stddevPitch)  # new
            mean_F0_list.append(meanF0)  # make a mean F0 list
            sd_F0_list.append(stdevF0)  # make a sd F0 list
            hnr_list.append(hnr)  # add HNR data

            # add raw jitter and shimmer measures
            localJitter_list.append(localJitter)
            localabsoluteJitter_list.append(localabsoluteJitter)
            rapJitter_list.append(rapJitter)
            ppq5Jitter_list.append(ppq5Jitter)
            ddpJitter_list.append(ddpJitter)
            localShimmer_list.append(localShimmer)
            localdbShimmer_list.append(localdbShimmer)
            apq3Shimmer_list.append(apq3Shimmer)
            aqpq5Shimmer_list.append(aqpq5Shimmer)
            apq11Shimmer_list.append(apq11Shimmer)
            ddaShimmer_list.append(ddaShimmer)

            # add the formant data
            f1_mean_list.append(f1_mean)
            f2_mean_list.append(f2_mean)
            f3_mean_list.append(f3_mean)
            f4_mean_list.append(f4_mean)
            f1_median_list.append(f1_median)
            f2_median_list.append(f2_median)
            f3_median_list.append(f3_median)
            f4_median_list.append(f4_median)

            count += 1

"""## This block of code adds all of that data we just generated to a Pandas data frame"""

# Add the data to Pandas
df = pd.DataFrame(np.column_stack([file_list, duration_list, meanIntensity_list, stddevIntensity_list, meanPitch_list, stddevPitch_list, mean_F0_list, sd_F0_list, hnr_list,
                                   localJitter_list, localabsoluteJitter_list, rapJitter_list,
                                   ppq5Jitter_list, ddpJitter_list, localShimmer_list,
                                   localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list,
                                   apq11Shimmer_list, ddaShimmer_list, f1_mean_list,
                                   f2_mean_list, f3_mean_list, f4_mean_list,
                                   f1_median_list, f2_median_list, f3_median_list,
                                   f4_median_list]),
                                   columns=['File_name', 'duration', 'meanIntesnity', 'stddevIntensity', 'meanPitch', 'stddevPitch', 'meanF0Hz', 'stdevF0Hz', 'HNR',
                                            'localJitter', 'localabsoluteJitter', 'rapJitter',
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer',
                                            'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer',
                                            'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean',
                                            'f3_mean', 'f4_mean', 'f1_median',
                                            'f2_median', 'f3_median', 'f4_median'])

# pcaData = runPCA(df) # Run jitter and shimmer PCA
# df = pd.concat([df, pcaData], axis=1)  # Add PCA data
# reload the data so it's all numbers
df.to_csv("csv_files/processed_results.csv", index=False)
df = pd.read_csv('csv_files_old/processed_results.csv', header=0)
df.sort_values('File_name').head(20)


# reload the data again
df.to_csv("csv_files/processed_results.csv", index=False)
df = pd.read_csv('csv_files_old/processed_results.csv', header=0)

df['fitch_vtl'] = ((1 * (35000 / (4 * df['f1_median']))) +
                   (3 * (35000 / (4 * df['f2_median']))) +
                   (5 * (35000 / (4 * df['f3_median']))) +
                   (7 * (35000 / (4 * df['f4_median'])))) / 4


# Write out the final dataframe
df.to_csv("csv_files/processed_results2.csv", index=False)

"""## Run this to tell you when it's done"""

print("finished")
