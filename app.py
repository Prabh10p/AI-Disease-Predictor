from flask import Flask, render_template, request
import pandas as pd
from src.Pipeline.predict_pipeline import Modelfeatures, Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle
import os

application = Flask(__name__)
app = application

# Load all symptom feature names
SYMPTOM_FEATURES = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain",
    "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
    "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm",
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck",
    "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance",
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_typhos", "depression", "irritability", "muscle_pain",
    "altered_sensorium", "red_spots_over_body", "belly_pain", "abnormal_menstruation",
    "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria",
    "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
    "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections",
    "coma", "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption",
    "fluid_overload_1", "blood_in_sputum", "prominent_veins_on_calf", "palpitations",
    "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze"
]

# Load LabelEncoder (for inverse_transform)
encoder_path = os.path.join("Artifacts", "label_encoder.pkl")
label_encoder = None
if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("form.html", symptoms=SYMPTOM_FEATURES)

    if request.method == "POST":
        try:
            # Get user input
            user_inputs = {feature: int(request.form.get(feature, 0)) for feature in SYMPTOM_FEATURES}
            input_df = pd.DataFrame([user_inputs])

            # Predict
            predictor = Pipeline()
            prediction = predictor.MakePipeline(input_df)

            # Convert prediction from label-encoded to original class name
            final_result = prediction[0]
            if label_encoder:
                final_result = label_encoder.inverse_transform([prediction[0]])[0]

            return render_template("result.html", symptoms=SYMPTOM_FEATURES, prediction=final_result)

        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=True)
