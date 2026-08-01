"""
app.py — Flask backend for Symptom-to-Disease Predictor.
Loads trained ML model and serves predictions via REST API.
"""
import os
import json
import math
import joblib
import urllib.request
import urllib.parse
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from utils import (
    preprocess_symptoms,
    fuzzy_match_symptoms,
    get_disease_info,
    check_red_flags,
    format_symptom_display,
)

import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR   = os.path.join(BASE_DIR, 'model')
MODEL_FILE  = os.path.join(MODEL_DIR, 'best_model.pkl')
ENCODER_FILE = os.path.join(MODEL_DIR, 'label_encoder.pkl')
FEATURES_FILE = os.path.join(MODEL_DIR, 'features.pkl')
METRICS_FILE  = os.path.join(MODEL_DIR, 'metrics.json')

# ─── Disease → Medical Specialty mapping ─────────────────────────────────────
DISEASE_SPECIALTY = {
    'Heart attack':                         ['cardiology', 'emergency', 'hospital'],
    'Hypertension':                         ['cardiology', 'hospital'],
    'Paralysis (brain hemorrhage)':         ['neurology', 'emergency', 'hospital'],
    'Migraine':                             ['neurology', 'hospital'],
    'Cervical spondylosis':                 ['orthopaedic', 'hospital'],
    'Osteoarthritis':                       ['orthopaedic', 'hospital'],
    'Arthritis':                            ['orthopaedic', 'hospital'],
    'Diabetes':                             ['endocrinology', 'hospital'],
    'Hypothyroidism':                       ['endocrinology', 'hospital'],
    'Hyperthyroidism':                      ['endocrinology', 'hospital'],
    'Hypoglycemia':                         ['endocrinology', 'hospital'],
    'Jaundice':                             ['gastroenterology', 'hospital'],
    'Hepatitis B':                          ['gastroenterology', 'hospital'],
    'Hepatitis C':                          ['gastroenterology', 'hospital'],
    'Hepatitis D':                          ['gastroenterology', 'hospital'],
    'Hepatitis E':                          ['gastroenterology', 'hospital'],
    'hepatitis A':                          ['gastroenterology', 'hospital'],
    'Alcoholic hepatitis':                  ['gastroenterology', 'hospital'],
    'Chronic cholestasis':                  ['gastroenterology', 'hospital'],
    'GERD':                                 ['gastroenterology', 'hospital'],
    'Peptic ulcer disease':                 ['gastroenterology', 'hospital'],
    'Gastroenteritis':                      ['hospital', 'clinic'],
    'Bronchial Asthma':                     ['pulmonology', 'hospital'],
    'Tuberculosis':                         ['pulmonology', 'hospital'],
    'Pneumonia':                            ['pulmonology', 'hospital'],
    'Malaria':                              ['hospital', 'clinic'],
    'Dengue':                               ['hospital', 'clinic'],
    'Typhoid':                              ['hospital', 'clinic'],
    'AIDS':                                 ['hospital'],
    'Chicken pox':                          ['clinic', 'hospital'],
    'Common Cold':                          ['clinic', 'hospital'],
    'Fungal infection':                     ['dermatology', 'clinic'],
    'Acne':                                 ['dermatology', 'clinic'],
    'Psoriasis':                            ['dermatology', 'clinic'],
    'Impetigo':                             ['dermatology', 'clinic'],
    'Allergy':                              ['clinic', 'hospital'],
    'Drug Reaction':                        ['hospital', 'clinic'],
    'Urinary tract infection':              ['urology', 'clinic', 'hospital'],
    'Dimorphic hemmorhoids(piles)':         ['gastroenterology', 'clinic'],
    'Varicose veins':                       ['hospital', 'clinic'],
    '(Vertigo) Paroxysmal Positional Vertigo': ['neurology', 'clinic'],
}

# ─── Load model artifacts ─────────────────────────────────────────────────────
print("Loading model artifacts...")
try:
    model          = joblib.load(MODEL_FILE)
    label_encoder  = joblib.load(ENCODER_FILE)
    feature_columns = joblib.load(FEATURES_FILE)

    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)

    MODEL_NAME     = metrics.get('best_model', 'Unknown')
    MODEL_ACCURACY = metrics.get('best_test_accuracy', 0.0)
    ALL_DISEASES   = metrics.get('diseases', [])
    MODEL_LOADED   = True
    print(f"  Model: {MODEL_NAME} (accuracy: {MODEL_ACCURACY:.2%})")
    print(f"  Features: {len(feature_columns)}, Diseases: {len(ALL_DISEASES)}")
except FileNotFoundError as e:
    print(f"  [WARNING] Model not found: {e}")
    print("  Run: python data/generate_dataset.py && python model/train.py")
    MODEL_LOADED   = False
    feature_columns = []
    label_encoder   = None
    model           = None
    MODEL_NAME      = 'Not trained yet'
    MODEL_ACCURACY  = 0.0
    ALL_DISEASES    = []

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))


@app.route('/')
def index():
    return render_template(
        'index.html',
        model_name=MODEL_NAME,
        model_accuracy=f"{MODEL_ACCURACY:.1%}",
        model_loaded=MODEL_LOADED,
        n_symptoms=len(feature_columns),
        n_diseases=len(ALL_DISEASES),
    )


@app.route('/google0881dea34fe83802.html')
def google_verification():
    return send_from_directory(BASE_DIR, 'google0881dea34fe83802.html')


@app.route('/robots.txt')
def robots():
    return send_from_directory(BASE_DIR, 'robots.txt')


@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(BASE_DIR, 'sitemap.xml')



@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return sorted list of all known symptoms in display format."""
    symptoms = [
        {'id': s, 'label': format_symptom_display(s)}
        for s in sorted(feature_columns)
    ]
    return jsonify({'symptoms': symptoms, 'count': len(symptoms)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept a list of symptoms and return top-3 disease predictions.
    Expects JSON: { "symptoms": ["symptom1", "symptom2", ...] }
    or: { "text": "free text describing symptoms" }
    """
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not trained yet. Please run training first.',
            'code': 'MODEL_NOT_READY'
        }), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400

    # ── Resolve input mode ───────────────────────────────────────────────────
    free_text_matched = []
    free_text_unmatched = []

    if 'text' in data and data['text']:
        # Free-text mode
        matched, unmatched = fuzzy_match_symptoms(data['text'], feature_columns)
        symptom_list = matched
        free_text_matched = matched
        free_text_unmatched = unmatched
    elif 'symptoms' in data:
        symptom_list = [s.strip() for s in data['symptoms'] if s.strip()]
    else:
        return jsonify({'error': 'Provide "symptoms" list or "text" field'}), 400

    # ── Validate minimum symptoms ────────────────────────────────────────────
    if len(symptom_list) < 3:
        return jsonify({
            'error': 'Please provide at least 3 symptoms for a reliable prediction.',
            'code': 'TOO_FEW_SYMPTOMS',
            'matched_so_far': [format_symptom_display(s) for s in symptom_list],
        }), 400

    # ── Preprocess ───────────────────────────────────────────────────────────
    feature_vector = preprocess_symptoms(symptom_list, feature_columns)
    X = np.array(feature_vector).reshape(1, -1)

    # ── Predict ──────────────────────────────────────────────────────────────
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:3]
        top_results = [
            {
                'disease': label_encoder.inverse_transform([i])[0],
                'confidence': float(proba[i]),
                'confidence_pct': f"{proba[i]*100:.1f}%",
            }
            for i in top_indices if proba[i] > 0.001
        ]
    else:
        # Fallback for models without predict_proba
        pred = model.predict(X)[0]
        top_results = [{
            'disease': label_encoder.inverse_transform([pred])[0],
            'confidence': 1.0,
            'confidence_pct': '100%',
        }]

    # ── Enrich with disease info ─────────────────────────────────────────────
    enriched = []
    for result in top_results:
        info = get_disease_info(result['disease'])
        enriched.append({
            **result,
            'description': info['description'],
            'precautions': info['precautions'],
            'see_doctor': info['see_doctor'],
            'severity': info['severity'],
        })

    # ── Red flag check ───────────────────────────────────────────────────────
    red_flags = check_red_flags(symptom_list)

    # ── Response ─────────────────────────────────────────────────────────────
    return jsonify({
        'predictions': enriched,
        'symptoms_used': [format_symptom_display(s) for s in symptom_list],
        'symptoms_raw': symptom_list,
        'red_flags': [format_symptom_display(s) for s in red_flags],
        'is_emergency': len(red_flags) > 0,
        'free_text_matched': [format_symptom_display(s) for s in free_text_matched],
        'free_text_unmatched': free_text_unmatched,
        'model_info': {
            'name': MODEL_NAME,
            'accuracy': MODEL_ACCURACY,
        },
        'disclaimer': (
            'This prediction is for educational purposes only and does NOT '
            'constitute medical advice or diagnosis. Always consult a licensed '
            'healthcare professional for medical concerns.'
        ),
    })


@app.route('/hospitals', methods=['POST'])
def find_hospitals():
    """
    Find hospitals near a given Indian PIN code, ranked by relevance to the disease.
    Expects JSON: { "pincode": "400001", "disease": "Malaria" }
    Uses Nominatim (geocoding) + Overpass API (hospital search) — both free/no key.
    """
    data = request.get_json(force=True)
    pincode  = str(data.get('pincode', '')).strip()
    disease  = str(data.get('disease', '')).strip()
    radius_m = int(data.get('radius_m', 10000))  # default 10 km

    if not pincode or len(pincode) != 6 or not pincode.isdigit():
        return jsonify({'error': 'Please enter a valid 6-digit Indian PIN code.'}), 400

    # ── Step 1: Geocode PIN code → lat/lng via Nominatim ─────────────────────
    try:
        nom_url = (
            'https://nominatim.openstreetmap.org/search?'
            + urllib.parse.urlencode({
                'postalcode': pincode,
                'country': 'India',
                'format': 'json',
                'limit': 1,
            })
        )
        req = urllib.request.Request(nom_url, headers={'User-Agent': 'MediPredict-AI/1.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            geo_results = json.loads(resp.read().decode())

        if not geo_results:
            return jsonify({'error': f'PIN code {pincode} not found. Please check and try again.'}), 404

        lat = float(geo_results[0]['lat'])
        lon = float(geo_results[0]['lon'])
        area_name = geo_results[0].get('display_name', pincode).split(',')[0]

    except Exception as e:
        return jsonify({'error': f'Could not geocode PIN code: {str(e)}'}), 502

    # ── Step 2: Query Overpass API for hospitals/clinics ──────────────────────
    try:
        overpass_query = f"""
        [out:json][timeout:15];
        (
          node["amenity"="hospital"](around:{radius_m},{lat},{lon});
          node["amenity"="clinic"](around:{radius_m},{lat},{lon});
          node["healthcare"="hospital"](around:{radius_m},{lat},{lon});
          way["amenity"="hospital"](around:{radius_m},{lat},{lon});
          way["amenity"="clinic"](around:{radius_m},{lat},{lon});
        );
        out center tags;
        """
        op_url = 'https://overpass-api.de/api/interpreter'
        req = urllib.request.Request(
            op_url,
            data=urllib.parse.urlencode({'data': overpass_query}).encode(),
            headers={'User-Agent': 'MediPredict-AI/1.0'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            op_data = json.loads(resp.read().decode())
        elements = op_data.get('elements', [])
    except Exception as e:
        return jsonify({'error': f'Hospital search failed: {str(e)}'}), 502

    # ── Step 3: Parse & enrich hospital data ─────────────────────────────────
    specialties = DISEASE_SPECIALTY.get(disease, ['hospital'])

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def relevance_score(tags):
        """Score hospital by specialty match (higher = more relevant)."""
        score = 0
        fields = ' '.join([
            tags.get('name', ''), tags.get('healthcare:speciality', ''),
            tags.get('description', ''), tags.get('speciality', ''),
        ]).lower()
        for sp in specialties:
            if sp in fields:
                score += 3
        if 'hospital' in tags.get('amenity', ''):
            score += 2
        if tags.get('emergency') == 'yes':
            score += 2
        if tags.get('name'):
            score += 1
        return score

    hospitals = []
    seen_names = set()
    for el in elements:
        tags = el.get('tags', {})
        name = tags.get('name', '').strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        el_lat = el.get('lat') or el.get('center', {}).get('lat')
        el_lon = el.get('lon') or el.get('center', {}).get('lon')
        if not el_lat or not el_lon:
            continue

        dist = haversine(lat, lon, float(el_lat), float(el_lon))
        hospitals.append({
            'name': name,
            'address': ', '.join(filter(None, [
                tags.get('addr:housename', ''),
                tags.get('addr:street', ''),
                tags.get('addr:suburb', ''),
                tags.get('addr:city', ''),
            ])) or tags.get('addr:full', 'Address not available'),
            'phone': tags.get('phone', tags.get('contact:phone', '')),
            'emergency': tags.get('emergency') == 'yes',
            'amenity': tags.get('amenity', 'hospital'),
            'speciality': tags.get('healthcare:speciality', tags.get('speciality', '')),
            'distance_m': round(dist),
            'distance_km': round(dist / 1000, 1),
            'relevance': relevance_score(tags),
            'maps_url': f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' ' + pincode)}",
            'lat': el_lat,
            'lon': el_lon,
        })

    # Sort: first by relevance (desc), then by distance (asc)
    hospitals.sort(key=lambda h: (-h['relevance'], h['distance_m']))
    top_hospitals = hospitals[:10]

    return jsonify({
        'hospitals': top_hospitals,
        'total_found': len(hospitals),
        'pincode': pincode,
        'area': area_name,
        'disease': disease,
        'specialties_searched': specialties,
        'radius_km': radius_m // 1000,
        'center': {'lat': lat, 'lon': lon},
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL_LOADED})


import threading
import webbrowser

if __name__ == '__main__':
    print("\nStarting Symptom-to-Disease Predictor...")
    print("   Opening http://127.0.0.1:5000 in your browser...\n")
    
    # Automatically open the browser after a 1 second delay
    threading.Timer(1.25, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    
    # Disable reloader so it doesn't open two tabs
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
