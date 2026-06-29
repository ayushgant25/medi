"""
Generate the Disease Prediction dataset programmatically.
This script creates training.csv and testing.csv files based on the
well-known Kaggle "Disease Prediction Using Machine Learning" dataset.
"""
import pandas as pd
import numpy as np
import os

# All 132 symptoms from the Kaggle dataset
ALL_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain',
    'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
    'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
    'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_stiffness',
    'movement_stiffness', 'swelling_joints', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze',
    'prognosis', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze',
    'prognosis', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching'
]

# Disease-symptom mapping (each disease maps to its primary symptoms)
DISEASE_SYMPTOMS = {
    'Fungal infection': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'],
    'Allergy': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
    'GERD': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough', 'chest_pain'],
    'Chronic cholestasis': ['itching', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain'],
    'Drug Reaction': ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_urination'],
    'Peptic ulcer disease': ['vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases', 'internal_itching'],
    'AIDS': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],
    'Diabetes': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level', 'increased_appetite', 'polyuria', 'blurred_and_distorted_vision'],
    'Gastroenteritis': ['vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea'],
    'Bronchial Asthma': ['fatigue', 'cough', 'high_fever', 'breathlessness', 'family_history', 'mucoid_sputum'],
    'Hypertension': ['headache', 'chest_pain', 'dizziness', 'loss_of_balance', 'lack_of_concentration'],
    'Migraine': ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision', 'excessive_hunger', 'stiff_neck', 'depression', 'irritability'],
    'Cervical spondylosis': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness', 'loss_of_balance'],
    'Paralysis (brain hemorrhage)': ['vomiting', 'headache', 'weakness_in_limbs', 'altered_sensorium'],
    'Jaundice': ['itching', 'vomiting', 'fatigue', 'weight_loss', 'high_fever', 'yellowish_skin', 'dark_urine', 'abdominal_pain'],
    'Malaria': ['chills', 'vomiting', 'high_fever', 'sweating', 'headache', 'nausea', 'diarrhoea', 'muscle_pain'],
    'Chicken pox': ['itching', 'skin_rash', 'fatigue', 'lethargy', 'high_fever', 'headache', 'loss_of_appetite', 'mild_fever', 'swelled_lymph_nodes', 'malaise', 'red_spots_over_body'],
    'Dengue': ['skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'malaise', 'muscle_pain', 'red_spots_over_body'],
    'Typhoid': ['chills', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look_(typhos)', 'belly_pain'],
    'hepatitis A': ['joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellowing_of_eyes', 'muscle_pain'],
    'Hepatitis B': ['itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine', 'loss_of_appetite', 'abdominal_pain', 'yellow_urine', 'yellowing_of_eyes', 'malaise', 'receiving_blood_transfusion', 'receiving_unsterile_injections'],
    'Hepatitis C': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'yellowing_of_eyes', 'family_history'],
    'Hepatitis D': ['joint_pain', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'yellowing_of_eyes'],
    'Hepatitis E': ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 'acute_liver_failure', 'coma', 'stomach_bleeding'],
    'Alcoholic hepatitis': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach', 'history_of_alcohol_consumption', 'fluid_overload.1', 'reeling'],
    'Tuberculosis': ['chills', 'vomiting', 'fatigue', 'weight_loss', 'cough', 'high_fever', 'sweating', 'loss_of_appetite', 'mild_fever', 'yellowing_of_eyes', 'phlegm', 'blood_in_sputum', 'rusty_sputum'],
    'Common Cold': ['continuous_sneezing', 'chills', 'fatigue', 'cough', 'headache', 'swelled_lymph_nodes', 'malaise', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'loss_of_smell', 'muscle_pain'],
    'Pneumonia': ['chills', 'fatigue', 'cough', 'high_fever', 'breathlessness', 'sweating', 'malaise', 'phlegm', 'chest_pain', 'fast_heart_rate', 'rusty_sputum'],
    'Dimorphic hemmorhoids(piles)': ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'],
    'Heart attack': ['vomiting', 'breathlessness', 'sweating', 'chest_pain'],
    'Varicose veins': ['fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf'],
    'Hypothyroidism': ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 'lethargy', 'dryness', 'hair_loss', 'brittle_nails', 'swelling_joints', 'depression', 'irritability', 'abnormal_menstruation'],
    'Hyperthyroidism': ['fatigue', 'mood_swings', 'weight_loss', 'restlessness', 'sweating', 'diarrhoea', 'fast_heart_rate', 'excessive_hunger', 'muscle_weakness', 'irritability', 'abnormal_menstruation'],
    'Hypoglycemia': ['vomiting', 'fatigue', 'anxiety', 'sweating', 'headache', 'nausea', 'blurred_and_distorted_vision', 'excessive_hunger', 'drying_and_tingling_lips', 'slurred_speech', 'irritability', 'palpitations'],
    'Osteoarthritis': ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain', 'swelling_joints', 'painful_walking'],
    'Arthritis': ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'painful_walking'],
    '(Vertigo) Paroxysmal Positional Vertigo': ['vomiting', 'headache', 'nausea', 'spinning_movements', 'loss_of_balance', 'unsteadiness'],
    'Acne': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
    'Urinary tract infection': ['burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine'],
    'Psoriasis': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'],
    'Impetigo': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'],
}

# Comprehensive symptom list (union of all symptoms used)
USED_SYMPTOMS = sorted(set(s for symptoms in DISEASE_SYMPTOMS.values() for s in symptoms))

def generate_dataset(n_samples_per_disease=120, noise_prob=0.05, seed=42):
    """Generate synthetic training data based on disease-symptom mappings."""
    np.random.seed(seed)
    rows = []
    
    for disease, core_symptoms in DISEASE_SYMPTOMS.items():
        for _ in range(n_samples_per_disease):
            row = {s: 0 for s in USED_SYMPTOMS}
            # Always include core symptoms (with small dropout)
            for s in core_symptoms:
                if s in USED_SYMPTOMS:
                    row[s] = 1 if np.random.random() > 0.05 else 0
            # Add a small amount of noise (random symptoms)
            for s in USED_SYMPTOMS:
                if row[s] == 0 and np.random.random() < noise_prob:
                    row[s] = 1
            row['prognosis'] = disease
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating training dataset...")
    train_df = generate_dataset(n_samples_per_disease=120)
    train_path = os.path.join(out_dir, 'training.csv')
    train_df.to_csv(train_path, index=False)
    print(f"  Saved {len(train_df)} rows to {train_path}")
    
    print("Generating testing dataset...")
    test_df = generate_dataset(n_samples_per_disease=30, seed=99)
    test_path = os.path.join(out_dir, 'testing.csv')
    test_df.to_csv(test_path, index=False)
    print(f"  Saved {len(test_df)} rows to {test_path}")
    
    print(f"\nDataset stats:")
    print(f"  Diseases: {train_df['prognosis'].nunique()}")
    print(f"  Symptoms (features): {len(USED_SYMPTOMS)}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Testing samples: {len(test_df)}")

if __name__ == '__main__':
    main()
