from flask import Flask, render_template, request
from predictor import Predictor

app = Flask(__name__, template_folder="templates")
predictor = Predictor()
predictor.load()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pred")
def pred():
    return render_template("pred.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict():
    if len(request.form) != 79:
        return render_template("result.html", prediction="Ada data yang belum diisi")
    
    # mengambil data dari form mulai dari Q1 sampai Q79 menggunakan for
    data = []
    for i in range(1, 80):
        data.append(int(request.form['Q'+str(i)]))

    prediction = predictor.predict(data)

    # konversi hasil prediksi
    if prediction == 0:
        pred = 'Cukup'
    elif prediction == 1:
        pred = 'Kurang Baik'
    elif prediction == 2:
        pred = 'Baik'
    elif prediction == 3:
        pred = 'Sangat Baik'

    # tampilkan hasil prediksi
    return render_template("result.html", prediction=pred)

#debug mode
if __name__ == "__main__":
    app.run(debug=True)
