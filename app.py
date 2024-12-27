from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and feature columns
model = joblib.load('models/maintenance_model_updated.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    machinery_type = request.form['machinery_type']
    temperature = float(request.form['temperature'])
    pressure = float(request.form['pressure'])
    vibration = float(request.form['vibration'])
    humidity = float(request.form['humidity'])
    wear_index = float(request.form['wear_index'])

    # Create the dataframe with the custom input values
    custom_data = {
        'Machinery Type': [machinery_type],
        'Temperature (°C)': [temperature],
        'Pressure (psi)': [pressure],
        'Vibration (mm/s)': [vibration],
        'Humidity (%)': [humidity],
        'Component Wear Index': [wear_index]
    }
    custom_df = pd.DataFrame(custom_data)

    # Apply one-hot encoding to 'Machinery Type' column
    custom_df_encoded = pd.get_dummies(custom_df, columns=['Machinery Type'], drop_first=True)

    # Reindex to match the training dataset structure
    custom_df_encoded = custom_df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Make prediction using the trained model
    predicted_score = model.predict(custom_df_encoded)

    # Return the result on a new page
    return render_template('result.html', predicted_score=predicted_score[0])
    
if __name__ == '__main__':
    app.run(debug=True)