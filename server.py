from flask import Flask, request, jsonify
from flask.helpers import send_from_directory
import pickle
import numpy as np
from flask_cors import CORS, cross_origin

app = Flask(__name__, static_folder='../dist', static_url_path='/')    
CORS(app)  # Enable CORS for all routes

# Load the saved model
loaded_model = pickle.load(open("./heart_model.sav", "rb"))

@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return jsonify({"message": "Welcome to the Heart Disease Prediction API"})

@app.route("/predict", methods=["GET"])
def printResult():
    return jsonify({"message": "Please send a POST request to this endpoint to get a prediction"})

@app.route("/heart", methods=["GET"])
@cross_origin()
def index():
    return jsonify({"message": "This is the heart disease predictor endpoint, use a POST request"})

@app.route("/lungs", methods=["GET"])
@cross_origin()
def index():
    return jsonify({"message": "This is the lung disease predictor endpoint, use a POST request"})

@app.route("/heart", methods=["POST"])
@cross_origin()

def predict():
    try:
        # Get the input data from the request
        input_data = request.json


        keys_in_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Create the tuple by converting values to appropriate data types
        input_data_tuple = tuple(
            int(input_data[key]) if key != 'oldpeak' else float(input_data[key])
            for key in keys_in_order
        )



        # Convert the input data to a numpy array and reshape it
        input_data_as_numpy_array = np.asarray(input_data_tuple)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


        # Make a prediction using the loaded model
        prediction = loaded_model.predict(input_data_reshaped)

        print(prediction[0])


        # Determine the result
        if prediction[0] == 0:
            result = "The person does not have a heart disease"
        else:
            result = "The person has a heart disease"


        return result

    except Exception as e:
        return jsonify({"error": str(e)})
    


@app.route("/lungs", methods=["POST"])
@cross_origin()
def predict_lung_cancer():
    try:
        # Get the input data from the request as a JSON
        input_data = request.json


      
        keys_in_order = [
        'age', 'gender', 'airPollution', 'alcoholUse', 'dustAllergy',
        'occupationalHazards', 'geneticRisk', 'chronicLungDisease',
        'balancedDiet', 'obesity', 'smoking', 'passiveSmoker',
        'chestPain', 'coughingOfBlood', 'fatigue', 'weightLoss',
        'shortnessOfBreath', 'wheezing', 'swallowingDifficulty',
        'clubbingOfFingerNails', 'frequentCold', 'dryCough', 'snoring'
        ];


        # print(input_data)

        input_data_tuple = tuple(
        int(input_data[key])
        for key in keys_in_order
    )

        # print(input_data_tuple)
      

        # Create the tuple by converting values to integer data type
    

        # input_data_tuple=(73,1,5,6,6,5,6,5,6,5,8,5,5,5,4,3,6,2,1,2,1,6,2)

        # Print the input data tuple (for debugging)
       

        # Load the trained lung cancer prediction model (make sure the path is correct)
        loaded_model = pickle.load(open("lung_model.sav", 'rb'))

        # Convert the input data tuple to a numpy array
        input_data_as_numpy_array = np.array(input_data_tuple, dtype=int)

        # Reshape the numpy array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make the lung cancer prediction
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            result = "The person does not have Lung Cancer"
        elif prediction[0] == 1:
            result = "The person could have lower levels of Lung Cancer"
        else:
            result ="The person could have higher levels of Lung Cancer"

        # Return the prediction result as JSON
        return result

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(debug=True)
