from flask import Flask, render_template, request
import pickle
import datetime

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_category = pickle.load(open("le_category.pkl", "rb"))
categories = pickle.load(open("categories.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", categories=categories)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        amt = float(request.form["amt"])

        category_text = request.form["category"]
        category = le_category.transform([category_text])[0]

        city_pop = float(request.form["city_pop"])

        dob = datetime.datetime.strptime(request.form["dob"], "%Y-%m-%d")
        age = datetime.datetime.now().year - dob.year

        hour = int(request.form["hour"])

        features = [[amt, category, city_pop, age, hour]]
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1] * 100

        status = "fraud" if prediction == 1 else "safe"

        return render_template("index.html",
                               categories=categories,
                               status=status,
                               fraud_prob=round(prob, 2))

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)