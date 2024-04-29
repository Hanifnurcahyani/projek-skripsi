import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn import model_selection, naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


def preprocessing():
        data_cleans = pd.read_excel('templates/data_cleans.xlsx')
        
        hasil = preprocessing()
        print(hasil)

    def testsplit(hasil):
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(hasil[0],hasil[1], test_size=0.2)
        labelencoder = LabelEncoder()
        Train_Y = labelencoder.fit_transform(Train_Y)
        Test_Y = labelencoder.fit_transform(Test_Y)
        return Train_Y, Test_Y, Train_X, Test_X
        
        hasil2 = testsplit(hasil)
        print(hasil2)

    def wordvectorization(hasil2):
        Tfidf_vect = TfidfVectorizer(max_features=2000)
        Tfidf_vect.fit(hasil[0])
        Test_X_Tfidf = Tfidf_vect.transform(hasil2)
        Test_X_Tfidf_fs = fs.transform(hasil3)
        return Train_X_Tfidf, Test_X_Tfidf

        hasil3 = wordvectorization(hasil2)
        print (hasil3)

        def featureselection(hasil3):
        fs = SelectKBest(score_func=mutual_info_classif, k=1800)
        fs.fit(hasil3[0], hasil2[0])
        Train_X_Tfidf_fs = fs.transform(hasil3[0])
        Test_X_Tfidf_fs = fs.transform(hasil3[1])
        return Train_X_Tfidf_fs, Test_X_Tfidf_fs

        hasil4 = featureselection(hasil3)
        print (hasil4)

        def classification(hasil4):
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(hasil4[0], hasil2[0])
        predictions_NB = Naive.predict(hasil4[1])
        # print("NaiveBayesAccuracyScore -> ", accuracy_score(predictions_NB, hasil2[1]) * 100)
        return "NaiveBayesAccuracyScore: ", accuracy_score(predictions_NB, hasil2[1]) * 100

        hasil5 = classification(hasil4)
        print (hasil5)