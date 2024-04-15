import pandas as pd

# Replace 'file1.csv' and 'file2.csv' with the actual file paths
file1_path = 'csv_files/participants_logs_csv.csv'
file2_path = 'csv_files/fixed_segments_features_with_all_features.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1_path)
df1.drop(columns=df1.columns[0], axis=1,  inplace=True)

df2 = pd.read_csv(file2_path)
df2.drop(columns=df2.columns[0], axis=1,  inplace=True)

df1 = df1.dropna()
df2 = df2.dropna()

# df2['File_name'] = df2['File_name'].str[:-18] + "'"

#df2['prefix'] = df2['File_name'].str[:-13]

df1['File_name'] = df1['File_name'].str.replace("exampleNewData", "sliced_examplefiles")
df1['File_name'] = df1['File_name'].str.lower()

df2['File_name'] = df2['File_name'].str[:-18] + "'"


print(df1)
print(df2)

print(df1.info())
print("\n\n\n")
print(df2.info())
print("\n\n\n\n\n")
# Merge the two DataFrames based on the "File_name" column
#merged_df = pd.merge(df1, df2, left_on='File_name', right_on=df2['File_name'].str[:-18], how="outer", suffixes=('_File1', '_File2'))

merged_df = pd.merge(df1, df2, on='File_name', how="outer", suffixes=('_File1', '_File2'))

merged_df = merged_df.dropna()

# Print or use the merged DataFrame
print(merged_df)
merged_df.to_csv("csv_files/segments_features_and_all_features_with_participants_logs.csv")
print(merged_df.info())
