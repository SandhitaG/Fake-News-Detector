from flask import Flask, request, jsonify
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('text')
    if not news_text:
        return jsonify({"error": "No text provided"}), 400

    vec = vectorizer.transform([news_text])
    prediction = model.predict(vec)[0]
    result = "Real" if prediction == 1 else "Fake"
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
