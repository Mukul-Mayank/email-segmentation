from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')  # Make sure you create this

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template('form.html', prediction_text=f"Prediction: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
