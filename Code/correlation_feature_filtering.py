import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example usage
csv_file = 'csv_files/segments_features_and_all_features_with_participants_logs_with_difficulty.csv'
# Read CSV into a pandas DataFrame
df = pd.read_csv(csv_file, skip_blank_lines=True)

categorical_features = ['WORD', 'COND', 'TASK']
for feature in categorical_features:
    df[feature] = df[feature].astype('category')
    df[feature] = df[feature].cat.codes

# Dropped for other reasons
df = df.drop(columns=['meanF0Hz', 'stdevF0Hz', 'File_name', 'TRLN', 'BLKN',
                      'TASK', 'TSKN', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])

# Compute correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(20, 12))
# Plot heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".1f")
plt.title('Correlation Heatmap')
plt.savefig(f'feature_correlation_heatmap')
plt.show()

remaining_features = list(df.columns)
corr_groups = []

count = 0
features_to_remain = []
for feature1 in remaining_features:
    my_corr = [feature1]
    for feature2 in remaining_features:
        if feature2 == feature1:
            continue
        if corr_matrix.loc[feature1, feature2] > 0.5:
            my_corr.append(feature2)
            remaining_features.remove(feature2)
    if len(my_corr) > 1:
        corr_groups.append(my_corr)
        count += len(my_corr)
        remaining_features.remove(feature1)

for feature in remaining_features:
    corr_groups.append([feature])
    count += 1

for group in corr_groups:
    print(group)

print(count)
print(len(list(df.columns)))
print(len(corr_groups))


csv_file = 'csv_files/segments_features_and_all_features_with_participants_logs_with_difficulty.csv'
# Read CSV into a pandas DataFrame
df = pd.read_csv(csv_file, skip_blank_lines=True)

# only keep f1_mean as a representative of the formants
df = df.drop(columns=['f2_mean', 'f3_mean', 'f4_mean', 'f1_median', 'f2_median', 'f3_median', 'f4_median', 'fitch_vtl'])

# only keep localJitter as a representative of the jitters and shimmers
df = df.drop(columns=['localabsoluteJitter', 'ppq5Jitter', 'localShimmer', 'apq3Shimmer', 'ddaShimmer',
                      'ddpJitter', 'rapJitter', 'apq11Shimmer', 'localdbShimmer', 'apq5Shimmer'])

# only keep Total_duration(s) as a representative of the jitters and shimmers
df = df.drop(columns=['duration'])

# drop difficulty because it has no purpose
df = df.drop(columns=['difficulty'])
df.to_csv("csv_files/data_csv_removed_high_correlation.csv")
print(df.info())

# plotting the new correlations
# Dropped for plotting reasons
df = df.drop(columns=['meanF0Hz', 'stdevF0Hz', 'File_name', 'TRLN', 'BLKN',
                      'TSKN', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])

categorical_features = ['WORD', 'COND', 'TASK']
for feature in categorical_features:
    df[feature] = df[feature].astype('category')
    df[feature] = df[feature].cat.codes

# Compute correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(20, 12))
# Plot heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".1f")
plt.title('Correlation Heatmap after removal')
plt.savefig(f'feature_correlation_heatmap_after_removal')
plt.show()


