# Human_Behavior_Language_Research_Project
Using ML tools to study the language learning process - Part of a research project in Haifa University

We received the recordings of multiple participants (about 50) saying different made up words (from 3 artificially constructed vocabularies) over the course of several days.
Then, We cleaned up the data (removed silent sections and removed unintelligible recordings), separated each recording to it's syllables, extracted auditory
features and then used the random forest classifier to find the distinguishing features of learning (thanks to sklearn's feature selection option we
were able to receive a ranking of each feature's importance in the construction of the random forest)
