import pandas as pd


def fixing_path_mistake(string):
    string = string.lower()
    indexes = [index for index in range(len(string))
               if string.startswith('sub', index)]

    string1 = string.replace(string[indexes[1] + 3: indexes[1] + 7], string[indexes[0] + 3: indexes[0] + 7])
    if string1 != string:
        print(string)
        print(string1)
        print('\n')

    return string1


file_name = 'segments_features_with_all_features.csv'

file_path = f'csv_files/{file_name}'

# Read the CSV files into DataFrames
df = pd.read_csv(file_path)

pd.set_option('max_colwidth', None)

df['File_name'] = df['File_name'].apply(fixing_path_mistake)

df.to_csv(f'csv_files/fixed_{file_name}')
