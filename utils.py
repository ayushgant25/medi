"""
utils.py — Helper functions for symptom preprocessing and NLP matching.
"""
import json
import os
import difflib
import re
from typing import List, Dict, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESC_FILE = os.path.join(BASE_DIR, 'data', 'symptom_descriptions.json')

# ─── Load disease info ────────────────────────────────────────────────────────
try:
    with open(DESC_FILE, 'r') as f:
        DISEASE_INFO = json.load(f)
except FileNotFoundError:
    DISEASE_INFO = {}

# ─── Red-flag symptoms that always trigger "seek immediate care" ──────────────
RED_FLAG_SYMPTOMS = {
    'chest_pain', 'breathlessness', 'fast_heart_rate', 'coma',
    'altered_sensorium', 'stomach_bleeding', 'blood_in_sputum',
    'acute_liver_failure', 'paralysis', 'loss_of_balance',
}

# ─── Synonym mapping: common English words → dataset symptom names ────────────
SYNONYM_MAP = {
    # fever / temperature
    'fever': 'high_fever', 'temperature': 'high_fever', 'high temp': 'high_fever',
    'mild fever': 'mild_fever', 'low grade fever': 'mild_fever',
    # pain
    'headache': 'headache', 'head pain': 'headache', 'head ache': 'headache',
    'stomach ache': 'stomach_pain', 'tummy ache': 'stomach_pain',
    'belly pain': 'belly_pain', 'abdominal pain': 'abdominal_pain',
    'back pain': 'back_pain', 'joint pain': 'joint_pain',
    'muscle pain': 'muscle_pain', 'chest pain': 'chest_pain',
    'neck pain': 'neck_stiffness',
    # digestive
    'nausea': 'nausea', 'vomiting': 'vomiting', 'vomit': 'vomiting',
    'throwing up': 'vomiting', 'diarrhea': 'diarrhoea', 'diarrhoea': 'diarrhoea',
    'loose stools': 'diarrhoea', 'indigestion': 'indigestion', 'heartburn': 'acidity',
    'constipation': 'constipation', 'bloating': 'passage_of_gases',
    'loss of appetite': 'loss_of_appetite', 'no appetite': 'loss_of_appetite',
    # skin
    'rash': 'skin_rash', 'itching': 'itching', 'itchy': 'itching',
    'hives': 'skin_rash', 'pimples': 'pus_filled_pimples',
    'skin peeling': 'skin_peeling', 'yellow skin': 'yellowish_skin',
    'yellowing': 'yellowish_skin', 'jaundice': 'yellowish_skin',
    # respiratory
    'cough': 'cough', 'coughing': 'cough', 'sneezing': 'continuous_sneezing',
    'runny nose': 'runny_nose', 'blocked nose': 'congestion',
    'sore throat': 'throat_irritation', 'phlegm': 'phlegm', 'mucus': 'phlegm',
    'shortness of breath': 'breathlessness', 'difficulty breathing': 'breathlessness',
    # eye / vision
    'blurry vision': 'blurred_and_distorted_vision', 'blurred vision': 'blurred_and_distorted_vision',
    'red eyes': 'redness_of_eyes', 'watery eyes': 'watering_from_eyes',
    'yellow eyes': 'yellowing_of_eyes',
    # general
    'fatigue': 'fatigue', 'tired': 'fatigue', 'weakness': 'fatigue',
    'lethargy': 'lethargy', 'malaise': 'malaise', 'dizzy': 'dizziness',
    'dizziness': 'dizziness', 'vertigo': 'spinning_movements',
    'chills': 'chills', 'shivering': 'shivering', 'sweating': 'sweating',
    'weight loss': 'weight_loss', 'weight gain': 'weight_gain',
    'anxiety': 'anxiety', 'depression': 'depression', 'mood swings': 'mood_swings',
    'dark urine': 'dark_urine', 'frequent urination': 'polyuria',
    'burning urination': 'burning_micturition', 'painful urination': 'burning_micturition',
    'swelling': 'swelling_joints', 'swollen': 'swelling_joints',
    'stiff neck': 'neck_stiffness', 'neck stiffness': 'neck_stiffness',
    'loss of smell': 'loss_of_smell', 'increased appetite': 'increased_appetite',
    'palpitations': 'palpitations', 'heart pounding': 'palpitations',
    'hair loss': 'hair_loss', 'cold hands': 'cold_hands_and_feets',
    'cold feet': 'cold_hands_and_feets',
    'excessive hunger': 'excessive_hunger', 'hungry': 'increased_appetite',
    'irritability': 'irritability', 'restlessness': 'restlessness',
}


def preprocess_symptoms(symptom_list: List[str], feature_columns: List[str]) -> List[int]:
    """
    Convert a list of symptom names into a binary feature vector
    aligned with the training feature columns.
    """
    # Normalize: lowercase, strip, replace spaces with underscores
    normalized = set()
    for s in symptom_list:
        s_clean = s.strip().lower().replace(' ', '_').replace('-', '_')
        normalized.add(s_clean)

    vector = [1 if col in normalized else 0 for col in feature_columns]
    return vector


def fuzzy_match_symptoms(text: str, known_symptoms: List[str]) -> Tuple[List[str], List[str]]:
    """
    Map free-text input to known symptom names.
    Returns (matched_symptoms, unmatched_tokens).
    Uses synonym dictionary first, then fuzzy matching as fallback.
    """
    text_lower = text.lower()
    matched = set()
    unmatched = []

    # 1. Check full synonym map first (multi-word phrases)
    for phrase, symptom in SYNONYM_MAP.items():
        if phrase in text_lower and symptom in known_symptoms:
            matched.add(symptom)

    # 2. Tokenize remaining text and try symptom lookup
    tokens = re.findall(r'\b[a-z][a-z_\s]{1,25}\b', text_lower)
    for token in tokens:
        token_clean = token.strip().replace(' ', '_')
        # Direct match
        if token_clean in known_symptoms:
            matched.add(token_clean)
            continue
        # Fuzzy match
        close = difflib.get_close_matches(token_clean, known_symptoms, n=1, cutoff=0.75)
        if close:
            matched.add(close[0])
        else:
            # Try synonym for token
            syn = SYNONYM_MAP.get(token.strip())
            if syn and syn in known_symptoms:
                matched.add(syn)
            else:
                if len(token.strip()) > 3:
                    unmatched.append(token.strip())

    return list(matched), unmatched


def get_disease_info(disease_name: str) -> Dict:
    """
    Retrieve disease description, precautions, and severity.
    Falls back to a generic response if disease not in database.
    """
    if disease_name in DISEASE_INFO:
        return DISEASE_INFO[disease_name]

    # Fuzzy lookup (handles slight name mismatches)
    close = difflib.get_close_matches(disease_name, DISEASE_INFO.keys(), n=1, cutoff=0.6)
    if close:
        return DISEASE_INFO[close[0]]

    return {
        'description': f'{disease_name} — detailed information not available in our database.',
        'precautions': ['Consult a licensed healthcare professional for guidance.'],
        'see_doctor': True,
        'severity': 'unknown',
    }


def check_red_flags(symptom_list: List[str]) -> List[str]:
    """
    Check if any submitted symptoms are in the red-flag list.
    Returns list of triggered red-flag symptoms.
    """
    return [s for s in symptom_list if s in RED_FLAG_SYMPTOMS]


def format_symptom_display(symptom: str) -> str:
    """Convert underscore symptom name to human-readable format."""
    return symptom.replace('_', ' ').title()
