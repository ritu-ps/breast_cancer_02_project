from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('cancer20.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Convert data into a DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)