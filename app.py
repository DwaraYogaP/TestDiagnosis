from flask import Flask, render_template, request
import joblib,os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
data_train = pd.read_excel("Book1.xlsx")

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(data_train['Gejala'])


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hasil', methods = ["POST"])
def predict():
    model = load("gigi_clf.pkl")

    array = request.form.getlist('check')
    jml_array = len(array)
    if (jml_array > 0):
        teks = ', '.join([str(elem) for elem in array])
        teks_vector = vectorizer.transform([teks])
        prediction = model.predict(teks_vector)
        return render_template("index.html", hasil_prediksi = prediction[0])
    else:
        return render_template("index.html", error = "Tidak ada masukan")
        

def load(file):
    load = joblib.load(open(os.path.join(file), "rb"))
    return load


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')