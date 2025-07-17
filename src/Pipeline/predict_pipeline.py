import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


import pandas as pd
import numpy as np
import dill
import os
import sys
from src.utils import load_object
from src.exception import CustomException

def clean_columns(df):
    # Strips whitespace, replaces spaces with _, and removes special characters
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)
    return df

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        try:
            # Step 1: Clean column names
            input_data = clean_columns(input_data)

            # Step 2: Load preprocessor and model
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Step 3: Transform and predict
            transformed_data = preprocessor.transform(input_data)
            preds = model.predict(transformed_data)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class Modelfeatures:
    def __init__(
        self,
        itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering,
        chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue, muscle_wasting,
        vomiting, burning_micturition, spotting_urination, fatigue, weight_gain,
        anxiety, cold_hands_and_feets, mood_swings, weight_loss, restlessness,
        lethargy, patches_in_throat, irregular_sugar_level, cough, high_fever,
        sunken_eyes, breathlessness, sweating, dehydration, indigestion, headache,
        yellowish_skin, dark_urine, nausea, loss_of_appetite, pain_behind_the_eyes,
        back_pain, constipation, abdominal_pain, diarrhoea, mild_fever, yellow_urine,
        yellowing_of_eyes, acute_liver_failure, fluid_overload, swelling_of_stomach,
        swelled_lymph_nodes, malaise, blurred_and_distorted_vision, phlegm,
        throat_irritation, redness_of_eyes, sinus_pressure, runny_nose, congestion,
        chest_pain, weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements,
        pain_in_anal_region, bloody_stool, irritation_in_anus, neck_pain, dizziness,
        cramps, bruising, obesity, swollen_legs, swollen_blood_vessels,
        puffy_face_and_eyes, enlarged_thyroid, brittle_nails, swollen_extremeties,
        excessive_hunger, extra_marital_contacts, drying_and_tingling_lips,
        slurred_speech, knee_pain, hip_joint_pain, muscle_weakness, stiff_neck,
        swelling_joints, movement_stiffness, spinning_movements, loss_of_balance,
        unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort,
        foul_smell_of_urine, continuous_feel_of_urine, passage_of_gases,
        internal_itching, toxic_look_typhos, depression, irritability, muscle_pain,
        altered_sensorium, red_spots_over_body, belly_pain, abnormal_menstruation,
        dischromic_patches, watering_from_eyes, increased_appetite, polyuria,
        family_history, mucoid_sputum, rusty_sputum, lack_of_concentration,
        visual_disturbances, receiving_blood_transfusion, receiving_unsterile_injections,
        coma, stomach_bleeding, distention_of_abdomen, history_of_alcohol_consumption,
        fluid_overload_1, blood_in_sputum, prominent_veins_on_calf, palpitations,
        painful_walking, pus_filled_pimples, blackheads, scurring, skin_peeling,
        silver_like_dusting, small_dents_in_nails, inflammatory_nails, blister,
        red_sore_around_nose, yellow_crust_ooze
    ):
        self.data = {
            'itching': itching,
            'skin_rash': skin_rash,
            'nodal_skin_eruptions': nodal_skin_eruptions,
            'continuous_sneezing': continuous_sneezing,
            'shivering': shivering,
            'chills': chills,
            'joint_pain': joint_pain,
            'stomach_pain': stomach_pain,
            'acidity': acidity,
            'ulcers_on_tongue': ulcers_on_tongue,
            'muscle_wasting': muscle_wasting,
            'vomiting': vomiting,
            'burning_micturition': burning_micturition,
            'spotting_ urination': spotting_urination,  # space added
            'spotting_urination': spotting_urination,
            'foul_smell_of_urine': foul_smell_of_urine,
            'dischromic_patches': dischromic_patches,
            'fluid_overload.1': fluid_overload_1,
            'toxic_look_(typhos)': toxic_look_typhos,
            'fatigue': fatigue,
            'weight_gain': weight_gain,
            'anxiety': anxiety,
            'cold_hands_and_feets': cold_hands_and_feets,
            'mood_swings': mood_swings,
            'weight_loss': weight_loss,
            'restlessness': restlessness,
            'lethargy': lethargy,
            'patches_in_throat': patches_in_throat,
            'irregular_sugar_level': irregular_sugar_level,
            'cough': cough,
            'high_fever': high_fever,
            'sunken_eyes': sunken_eyes,
            'breathlessness': breathlessness,
            'sweating': sweating,
            'dehydration': dehydration,
            'indigestion': indigestion,
            'headache': headache,
            'yellowish_skin': yellowish_skin,
            'dark_urine': dark_urine,
            'nausea': nausea,
            'loss_of_appetite': loss_of_appetite,
            'pain_behind_the_eyes': pain_behind_the_eyes,
            'back_pain': back_pain,
            'constipation': constipation,
            'abdominal_pain': abdominal_pain,
            'diarrhoea': diarrhoea,
            'mild_fever': mild_fever,
            'yellow_urine': yellow_urine,
            'yellowing_of_eyes': yellowing_of_eyes,
            'acute_liver_failure': acute_liver_failure,
            'fluid_overload': fluid_overload,
            'swelling_of_stomach': swelling_of_stomach,
            'swelled_lymph_nodes': swelled_lymph_nodes,
            'malaise': malaise,
            'blurred_and_distorted_vision': blurred_and_distorted_vision,
            'phlegm': phlegm,
            'throat_irritation': throat_irritation,
            'redness_of_eyes': redness_of_eyes,
            'sinus_pressure': sinus_pressure,
            'runny_nose': runny_nose,
            'congestion': congestion,
            'chest_pain': chest_pain,
            'weakness_in_limbs': weakness_in_limbs,
            'fast_heart_rate': fast_heart_rate,
            'pain_during_bowel_movements': pain_during_bowel_movements,
            'pain_in_anal_region': pain_in_anal_region,
            'bloody_stool': bloody_stool,
            'irritation_in_anus': irritation_in_anus,
            'neck_pain': neck_pain,
            'dizziness': dizziness,
            'cramps': cramps,
            'bruising': bruising,
            'obesity': obesity,
            'swollen_legs': swollen_legs,
            'swollen_blood_vessels': swollen_blood_vessels,
            'puffy_face_and_eyes': puffy_face_and_eyes,
            'enlarged_thyroid': enlarged_thyroid,
            'brittle_nails': brittle_nails,
            'swollen_extremeties': swollen_extremeties,
            'excessive_hunger': excessive_hunger,
            'extra_marital_contacts': extra_marital_contacts,
            'drying_and_tingling_lips': drying_and_tingling_lips,
            'slurred_speech': slurred_speech,
            'knee_pain': knee_pain,
            'hip_joint_pain': hip_joint_pain,
            'muscle_weakness': muscle_weakness,
            'stiff_neck': stiff_neck,
            'swelling_joints': swelling_joints,
            'movement_stiffness': movement_stiffness,
            'spinning_movements': spinning_movements,
            'loss_of_balance': loss_of_balance,
            'unsteadiness': unsteadiness,
            'weakness_of_one_body_side': weakness_of_one_body_side,
            'loss_of_smell': loss_of_smell,
            'bladder_discomfort': bladder_discomfort,
            'foul_smell_of urine': foul_smell_of_urine,  # space added
            'continuous_feel_of_urine': continuous_feel_of_urine,
            'passage_of_gases': passage_of_gases,
            'internal_itching': internal_itching,
            'toxic_look_(typhos)': toxic_look_typhos,  # parentheses and underscore restored
            'depression': depression,
            'irritability': irritability,
            'muscle_pain': muscle_pain,
            'altered_sensorium': altered_sensorium,
            'red_spots_over_body': red_spots_over_body,
            'belly_pain': belly_pain,
            'abnormal_menstruation': abnormal_menstruation,
            'dischromic _patches': dischromic_patches,  # space added
            'watering_from_eyes': watering_from_eyes,
            'increased_appetite': increased_appetite,
            'polyuria': polyuria,
            'family_history': family_history,
            'mucoid_sputum': mucoid_sputum,
            'rusty_sputum': rusty_sputum,
            'lack_of_concentration': lack_of_concentration,
            'visual_disturbances': visual_disturbances,
            'receiving_blood_transfusion': receiving_blood_transfusion,
            'receiving_unsterile_injections': receiving_unsterile_injections,
            'coma': coma,
            'stomach_bleeding': stomach_bleeding,
            'distention_of_abdomen': distention_of_abdomen,
            'history_of_alcohol_consumption': history_of_alcohol_consumption,
            'fluid_overload.1': fluid_overload_1,  # dot restored
            'blood_in_sputum': blood_in_sputum,
            'prominent_veins_on_calf': prominent_veins_on_calf,
            'palpitations': palpitations,
            'painful_walking': painful_walking,
            'pus_filled_pimples': pus_filled_pimples,
            'blackheads': blackheads,
            'scurring': scurring,
            'skin_peeling': skin_peeling,
            'silver_like_dusting': silver_like_dusting,
            'small_dents_in_nails': small_dents_in_nails,
            'inflammatory_nails': inflammatory_nails,
            'blister': blister,
            'red_sore_around_nose': red_sore_around_nose,
            'yellow_crust_ooze': yellow_crust_ooze
        }

    def to_dataframe(self):
        return pd.DataFrame([self.data])
