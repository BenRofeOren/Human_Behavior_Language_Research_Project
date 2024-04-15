import mpld3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_format_list(list_to_print):
    res = '['
    for value in list_to_print:
        res += f'{value:.3f}, '
    if len(res) > 3:
        res = res[:-2]
    res += ']'
    return res


def print_format_dict(dict_to_print, reverse_order=False):
    res = '{'
    keys = list(dict_to_print.keys())
    if reverse_order:
        keys = keys[::-1]
    for key in keys:
        res += f'{key}: {dict_to_print[key]:.3f}, '
    if len(res) > 3:
        res = res[:-2]
    res += '}'
    return res


def normalize_per_feature(df, features):
    df_copy = df.copy()
    means = df[features].mean()
    for feature in features:
        df_copy[feature] = (df_copy[feature] - means[feature]) / means[feature]
    return df_copy


def normalize_per_subject(df, features):
    df_copy = df.copy()
    grouped_subject = df_copy.groupby('SUBJ')
    for subject, subject_group in grouped_subject:
        for feature in features:
            mean = subject_group[feature].mean()
            subject_group[feature] = (subject_group[feature] - mean) / mean
        df_copy.loc[subject_group.index, features] = subject_group[features]
    return df_copy


def normalize_per_subject_per_session(df, features):
    df_copy = df.copy()
    grouped_subject = df_copy.groupby('SUBJ')
    for subject, subject_group in grouped_subject:
        grouped_session = subject_group.groupby('SESS')
        for session, session_group in grouped_session:
            for feature in features:
                mean = session_group[feature].mean()
                session_group[feature] = (session_group[feature] - mean) / mean
            df_copy.loc[session_group.index, features] = session_group[features]

    return df_copy


def run_random_forest_with_feature_selection(X_train, y_train, X_test, y_test, output_addition,
                                             k_folds=3, n_estimators=100, num_selected_features=0,
                                             max_tree_depth=None):
    # Initialize Random Forest classifier
    if num_selected_features == 0:
        num_selected_features = len(X_train.columns)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_tree_depth)

    # Use SelectFromModel for feature selection
    feature_selector = SelectFromModel(rf, max_features=num_selected_features)
    X_selected = feature_selector.fit_transform(X_train, y_train)

    # Get selected feature names and their importance scores
    selected_feature_names = X_train.columns[feature_selector.get_support()]
    feature_importance_scores = feature_selector.estimator_.feature_importances_

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_selected, y_train, cv=kf)

    mean_accuracy = cv_scores.mean()

    # Print the cross-validation scores
    output_addition.append(f"Cross-validation scores: {print_format_list(cv_scores)}")
    print(output_addition[-1])
    output_addition.append(f"Mean CV accuracy over train set: {mean_accuracy:.3f}")
    print(output_addition[-1])

    rf.fit(X_train, y_train)

    y_pred_test = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    output_addition.append(f"Accuracy of model on test set: {test_accuracy:.3f}")
    print(output_addition[-1])
    c_matrix = confusion_matrix(y_test, y_pred_test)
    c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]

    # Print selected feature names and their importance scores
    dict_selected_feature = {}
    output_addition.append(f"\nSelected Features and Importance Scores:")
    print(output_addition[-1])
    for name, score in zip(selected_feature_names, feature_importance_scores):
        dict_selected_feature[name] = score

    dict_selected_feature = {k: v for k, v in sorted(dict_selected_feature.items(), key=lambda item: item[1])}
    returned_features = [k for k, v in sorted(dict_selected_feature.items(), key=lambda item: item[1])]

    output_addition.append(f"{print_format_dict(dict_selected_feature, reverse_order=True)}")
    print(output_addition[-1])

    return returned_features, c_matrix


def plot_feature_boxplot(df_all_classes, important_features, title, target_label):
    sns.set(style="ticks")
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # normalizing all features, so they fit in
    # the same range for visibility considerations
    for f in important_features:
        df_all_classes[f] = (df_all_classes[f] - df_all_classes[f].mean()) / df_all_classes[f].std()

    df_melted = pd.melt(df_all_classes, id_vars=[target_label], value_vars=important_features)
    # Plot the boxplot using seaborn
    p = sns.boxplot(x="variable", y="value", hue=target_label, data=df_melted, palette="Pastel1", showfliers=False)
    labels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticklabels([label[::-1] if not label.isascii() else label for label in labels])
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)
    p.set_title(title)
    # plt.show()
    return fig


def make_report(config):
    path_to_data = config['PATH_TO_DATA']
    label_column = config['label_column']
    possible_values = config['possible_values']
    remove_session_f = config['remove_session_2_3']
    smote_flag = config['smote_flag']
    normalization_method = config['normalization_method']
    features_to_normalize = config['features_to_normalize']
    junk_features = config['junk_features']
    unfair_features = config['unfair_features']
    dropped_features = config['dropped_features']
    train_test_split_rate = config['train_test_split_rate']
    no_of_estimators = config['no_of_estimators']
    max_tree_depth = config['max_tree_depth']
    no_of_k_folds = config['no_of_k_folds']

    smote_string = ''
    if not config['smote_flag']:
        smote_string += 'out'

    corr_string = ''
    if 'correlation' not in path_to_data:
        corr_string += 'out'

    results_directory = f'reports/pred_{label_column}_no_separation' \
                        f'_with{smote_string}_SMOTE_' \
                        f'normalization_{normalization_method}' \
                        f'_with_{no_of_k_folds}_fold' \
                        f'_with{corr_string}_rm_high_corr'
    if remove_session_f:
        results_directory += '_remove_SESS_2_3'

    os.mkdir(results_directory)
    # Save the configuration to a JSON file
    with open(f"{results_directory}/config.json", "w") as json_file:
        json.dump(config, json_file, indent=4)

    df = pd.read_csv(path_to_data)

    if 'correlation' in path_to_data:
        df = df.drop(columns=['Unnamed: 0.3'])
        new_features_to_normalize = []
        for feature in features_to_normalize:
            if feature in df.columns:
                new_features_to_normalize.append(feature)
        features_to_normalize = new_features_to_normalize

    if remove_session_f:
        df = df.drop(df[df['SESS'] == 2].index)
        df = df.drop(df[df['SESS'] == 3].index)

    if normalization_method == "per_feature":
        df = normalize_per_feature(df, features_to_normalize)
    elif normalization_method == "per_subject":
        df = normalize_per_subject(df, features_to_normalize)
    elif normalization_method == "per_subject_per_session":
        df = normalize_per_subject_per_session(df, features_to_normalize)
    elif normalization_method == "drop":
        df = df.drop(columns=features_to_normalize)

    if label_column == 'ACCR':
        df = df.drop(columns=['SESS'])

    # Dropped for being junk
    df = df.drop(columns=junk_features)

    # Dropped for being unfair or duplicates of other features
    df = df.drop(columns=unfair_features)

    # Dropped for other reasons
    df = df.drop(columns=dropped_features)

    categorical_features = ['WORD', 'COND']

    if 'TASK' not in unfair_features:
        categorical_features.append('TASK')

    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
        df[feature] = df[feature].cat.codes

    plt.figure(figsize=(10, 6))
    # Iterate through the groups and print them (or perform any desired operation)
    lines_to_save = []

    X, y = df.drop(columns=label_column), df[label_column]

    # it's SMOTE-ing time
    if smote_flag:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)

    print(X.info(), y.info())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_rate)

    X_train_copy = X_train
    y_train_copy = y_train
    X_test_copy = X_test
    y_test_copy = y_test

    remove_each_time = int(len(X_train_copy.columns) * 0.3)
    accuracy_over_time = []

    i = 0
    plt.figure(figsize=(10, 6))
    most_important_features = []

    while 5 < len(X_train_copy.columns):
        no_of_remaining = len(X_train_copy.columns)
        lines_to_save.append(f"no. of remaining features: {no_of_remaining}")
        print(lines_to_save[-1])
        lines_to_save.append(f"remaining features: {X_train_copy.columns.to_list()}")
        print(lines_to_save[-1])
        important_features, matrix = run_random_forest_with_feature_selection(X_train_copy, y_train_copy,
                                                                              X_test_copy, y_test_copy,
                                                                              lines_to_save,
                                                                              n_estimators=no_of_estimators,
                                                                              k_folds=no_of_k_folds,
                                                                              max_tree_depth=max_tree_depth)
        accuracy_over_time.append(matrix)

        important_features = important_features[::-1]  # important features in descending order
        lines_to_save.append(f"important features in descending order: {important_features}")
        print(lines_to_save[-1])
        for feature in X_train_copy.columns:
            if feature not in important_features:
                important_features.append(feature)

        lines_to_save.append(f"cut features: {important_features[remove_each_time:]}")
        print(lines_to_save[-1])
        lines_to_save.append(f"\n")
        print(lines_to_save[-1])

        if len(important_features) - remove_each_time > 5:
            important_features = important_features[:-remove_each_time]
            remove_each_time = int(max(len(X_train_copy.columns) * 0.3, 6))
        else:
            features_to_keep = min(len(important_features), 5)
            important_features = important_features[:features_to_keep]

        most_important_features = important_features

        X_train_copy = X_train_copy[important_features]
        X_test_copy = X_test_copy[important_features]

        # Build the plot
        plt.subplot(321 + i)
        sns.set(font_scale=1.4)
        sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                    cmap=plt.cm.Greens, linewidths=0.2)

        # Add labels to the plot
        class_names = possible_values
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks2, class_names, rotation=25)
        plt.yticks(tick_marks2, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Iteration_{i}_with_{no_of_remaining}_features')

        i = i + 1

    plt.tight_layout(pad=2)
    plt.savefig(f'{results_directory}/confusion_matrix')

    # Plot the 5 most important features in descending order in a box plot
    fig = plot_feature_boxplot(df, important_features=most_important_features,
                               title="Box Plot of Important Features", target_label=label_column)
    plt.savefig(f'{results_directory}/box_plot_important_features')
    mpld3.save_html(fig, f'{results_directory}/box_plot_important_features.html')

    # Open a text file in write mode
    with open(f'{results_directory}/results.txt', "w") as file:
        # Write each string to the file
        for string in lines_to_save:
            file.write(string + "\n")


# --- Configurations ---
config = {
    "PATH_TO_DATA": 'csv_files/segments_features_and_all_features_with_participants_logs_with_difficulty.csv',

    'label_column': 'ACCR',  # what we want to predict
    'possible_values': ['0', '1'],  # possible values for label column

    'remove_session_2_3': True,

    'group_by_feature': None,  # separate data according to this feature
    'smote_flag': True,  # use smote for data augmentation or not

    'normalization_method': '',  # how to normalize data
    'features_to_normalize': ['meanIntesnity', 'stddevIntensity',
                              'meanPitch', 'stddevPitch', 'HNR','s_last_pitch_std',
                              'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter',
                              'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer',
                              'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'f1_mean', 'f2_mean',
                              'f3_mean', 'f4_mean', 'f1_median', 'f2_median', 'f3_median',
                              'f4_median', 'JitterPCA', 'ShimmerPCA', 'fitch_vtl', 's_last_pitch_mean',
                              's_first_intensity_mean', 's_first_intensity_std', 's_first_pitch_mean',
                              's_first_pitch_std', 's_last_intensity_mean', 's_last_intensity_std'],
    'features_remove_outliers': [],

    'junk_features': ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'],
    # 'unfair_features': ['meanF0Hz', 'stdevF0Hz', 'File_name', 'SUBJ', 'SESS', 'TRLN', 'BLKN', 'ACCR',
    #                     'Reaction_time(s)', 'TASK'],
    'unfair_features': ['meanF0Hz', 'stdevF0Hz', 'File_name', 'SUBJ', 'TRLN', 'BLKN', 'ACCR',
                        'Reaction_time(s)', 'TASK', 'TSKN'],

    'dropped_features': [],

    'train_test_split_rate': 0.3,

    'no_of_estimators': 100,
    'max_tree_depth': None,  # put None for maximum depth
    'no_of_k_folds': 5
}
# ----------------------

# make_report(config)

# ---------------------------------------------- #

file_options = ['csv_files/data_csv_removed_high_correlation.csv',
                'csv_files/segments_features_and_all_features_with_participants_logs_with_difficulty.csv']

predict_options = ['SESS']
remove_session_options = [True]

predict_possible_values = [[1, 4]]
normalization_methods = ['unchanged', 'per_feature', 'per_subject', 'drop']
smote = [True]

j = 0

for remove_session_flag in remove_session_options:
    config['remove_session_2_3'] = remove_session_flag
    for file_name in file_options:
        config['PATH_TO_DATA'] = file_name
        for i in range(len(predict_options)):
            config['label_column'] = predict_options[i]
            config['possible_values'] = predict_possible_values[i]
            for normalization_method in normalization_methods:
                config['normalization_method'] = normalization_method
                for s in smote:
                    config['smote_flag'] = s
                    make_report(config)



