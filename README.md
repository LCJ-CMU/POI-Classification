# POI-Classification 
The POI dataset used is from data.world.[1] POI (point of interest) contains information/description of multiple geographical locations (name, coordinate, category...)
The dataset contains (name, category) info. of 13004 geographical locations from 10 categories. We implement M-ary classification on the dataset only based on the name of locations. The project is contains data preprocessing, feature extraction, classification and evaluation.


![Confusion Matrix](https://github.com/LCJ-CMU/POI-Classification/blob/main/result/word%20frequency.png)


Data preprocessing: tokenizing text into words, upsampling and downsampling to obtain balanced dataset.

Feature extraction: word2vec by pre-trained NLP model (embedding)[2]; Bag-of-Words

Classification: Adaboost Classification, Random Forest, Support Vector Machine

Evaluation: Best Trial: SVM with word embedding feature: total accuracy = 79% - single class accuary>50%
![Confusion Matrix](https://github.com/LCJ-CMU/POI-Classification/blob/main/result/cm_svm_e.png)


Suggestion: Name feature is not adequate for POI classification, names may not suggest information on category classification, for example: {Hilliard, Industrial}

[1] data.world. (2018, May 18). Points of interest - dataset by smartcolumbusos. data.world. Retrieved May 6, 2023, from https://data.world/smartcolumbusos/5d869207-e885-4f50-aecc-7ee1a7dd9e40 

[2] Google. (2017, December 11). GoogleNews-vectors-negative300. Kaggle. Retrieved May 6, 2023, from https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300 
