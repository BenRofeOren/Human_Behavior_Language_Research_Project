import os
import shutil
import numpy as np
import os
import pandas as pd
import re
from unidecode import unidecode  # for removing special characters like é


def remove_special_annotations(word):
    return str(unidecode(str(word)).lower())


def create_File_name(row):
    return repr(f'exampleNewData\\Sub0{row["SUBJ"]}\\S{row["SESS"]}\\{row["TASK"]}{row["TSKN"]}\\'
                f'Sub0{row["SUBJ"]}_Block{row["BLKN"]}_Trial_{row["TRLN"]:03}_({row["WORD"]})_')


def process_dataframe(df):
    important_features = df[['SUBJ', 'SESS', 'TSKN', 'BLKN', 'ACCR', "TASK", "TRLN", "WORD", "COND"]]

    important_features['WORD'] = df['WORD'].apply(remove_special_annotations)

    #print('exampleNewData\\Sub0' + str(df["SUBJ"]))

    important_features["File_name"] = important_features.apply(create_File_name, axis=1)

    #print(important_features.head().to_string())

    return important_features


source_directory = "participants_logs"
destination_directory = "participants_logs_csv_files"

participants_logs_converted = []

count = 0
# Walk through all directories and files in the directory tree
for root, dirs, files in os.walk(source_directory):
    for filename in files:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):  # Adjust file extensions as needed
            if count % 100 == 0:
                print("processed ", count, " files")
            file_path = os.path.join(root, filename)

            # Read all sheets from the Excel file into a dictionary of dataframes
            excel_sheets = pd.read_excel(file_path, sheet_name=None)

            #sheets_encountered = 0
            # Iterate over all sheets in the dictionary
            for sheet_name, df in excel_sheets.items():
                # Call the custom function on each dataframe
                if "_test_out" in sheet_name or "train_ou" in sheet_name:  # this needs to be "train_ou" because the
                                                                          # sheets are named "...train_ou"
                    print(sheet_name)
                    print(df.keys())
                    print()
                    important_features = process_dataframe(df)
                    participants_logs_converted.append(important_features)
                    #sheets_encountered += 1
            count += 1


participants_logs_csv = pd.concat(participants_logs_converted, axis=0)
participants_logs_csv['File_name'] = participants_logs_csv['File_name'].str.replace(' ', '')

#participants_logs_csv.drop(columns=participants_logs_csv.columns[0], axis=1, inplace=True)
participants_logs_csv['File_name'] = participants_logs_csv['File_name'].str.replace('.0', '')

print(participants_logs_csv)
print(participants_logs_csv.info())

participants_logs_csv.to_csv("csv_files/participants_logs_csv.csv")

