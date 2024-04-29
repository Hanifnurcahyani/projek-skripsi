from flask import Flask, render_template
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
from sklearn.metrics import accuracy_score

app = Flask(__name__)

#@app.route("/")
#def home():
    #return "<p>Hanif</p>"

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

#@app.route("/dataset_ori")
#def dataset_ori():
    #dataset_ori = pd.read_csv('templates/dataset_ori.csv', error_bad_lines=False, delimiter=',',
    #header=0)
    #dataset_ori = np.array(dataset_ori)
    #review = dataset_ori[:, 0]
    #sentiment = dataset_ori[:, 1]
    #return render_template('dataset_ori.html', review=review, sentiment=sentiment)

@app.route("/data_cleans")
def data_cleans():
    data_cleans = pd.read_csv('templates/data_cleans.csv', error_bad_lines=False, delimiter=',',
    header=0)
    data_cleans = np.array(data_cleans)
    review = data_cleans[:, 0]
    sentiment = data_cleans[:, 1]
    return render_template('data_cleans.html', review=review, sentiment=sentiment)

@app.route("/akurasi")
def akurasi():
    def load_data():
        data_cleans = pd.read_excel('templates/data_cleans.xlsx')
        return data_cleans

    imdb_data = load_data()
    df = pd.DataFrame(imdb_data[['review','sentiment']])
    data_cleans = load_data()

    #tfidf
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data_cleans['review'],data_cleans['sentiment'], test_size=0.2)
    
    Tfidf_vect = TfidfVectorizer(max_features=1800)
    Tfidf_vect.fit(data_cleans['review'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
 
    print(Tfidf_vect.vocabulary_)
    print(Train_X_Tfidf)

    #information gain
    fs = SelectKBest(score_func=mutual_info_classif, k=1800)
    fs.fit(Train_X_Tfidf, Train_Y)
    Train_X_Tfidf_fs = fs.transform(Train_X_Tfidf)
    Test_X_Tfidf_fs = fs.transform(Test_X_Tfidf)

    #klasifikasi
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf_fs, Train_Y)
    predictions_NB = Naive.predict(Test_X_Tfidf_fs)
    akurasiNB=(accuracy_score(predictions_NB, Test_Y)*100)
    print("Naive Bayes Accuracy Score -> ", akurasiNB)

    #Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=400, learning_rate=0.5, random_state=0)
    #Train Adaboost Classifer
    model1 = abc.fit(Train_X_Tfidf_fs, Train_Y)
    #Predict the response for test dataset
    y_pred = model1.predict(Test_X_Tfidf_fs)
    akurasiAB=(accuracy_score(predictions_NB, y_pred)*100)
    print("AdaBoost Classifier Model Accuracy:", akurasiAB)
    return render_template('akurasi.html', acc_NB=akurasiNB, acc_MNBIG=akurasiAB)

@app.route("/akurasi2")
def akurasi2():
    return render_template('akurasi2.html')

@app.route("/datasetup")
def datasetup():
    return render_template('datasetup.html')

@app.route("/prediksi",methods=['GET','POST'])
def prediksi():
    from flask import request
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentimen = ''
    Tfidf_vect = TfidfVectorizer(max_features=1800)

    def load_data():
        data_cleans = pd.read_excel('templates/data_cleans.xlsx')
        return data_cleans

    imdb_data = load_data()
    df = pd.DataFrame(imdb_data[['review','sentiment']])
    data_cleans = load_data()

    Tfidf_vect = TfidfVectorizer(max_features=1800)
    Tfidf_vect.fit(data_cleans['review'])

    if request.method ==  "POST":

        projectpath = request.form.get('projectFilepath', None)
        tfidf_vektor = Tfidf_vect.transform([projectpath])
        loaded_model = joblib.load("model3.joblib")
        pred = loaded_model.predict(tfidf_vektor)
        if pred == 1:
            sentimen = 'positif'
        else:
            sentimen = 'negatif'
       
      
    return render_template('predict.html', hasil_sentiment=sentimen)

if __name__ == '__main__':
    app.run(debug=True)