# ============================================================
#  APPLICATION WEB - Prediction du niveau d'anxiete etudiant
#  Framework : Flask | Dataset : Student Mental Health (reel)
# ============================================================
# Pour lancer : python app.py
# Ouvrir dans le navigateur : http://localhost:5000
# ============================================================

from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import os
import sqlite3
import json

app = Flask(__name__)

OUTPUTS = "outputs"

def load_artifacts():
    model     = joblib.load(f"{OUTPUTS}/best_model.pkl")
    scaler    = joblib.load(f"{OUTPUTS}/scaler.pkl")
    features  = joblib.load(f"{OUTPUTS}/feature_names.pkl")
    le_course = joblib.load(f"{OUTPUTS}/course_encoder.pkl")
    with open(f"{OUTPUTS}/metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, features, le_course, metrics

try:
    model, scaler, FEATURE_NAMES, le_course, METRICS = load_artifacts()
    print("OK Artefacts charges avec succes")
    print(f"   Features attendues : {FEATURE_NAMES}")
except Exception as e:
    print(f"ERREUR de chargement : {e}")
    model = scaler = FEATURE_NAMES = le_course = METRICS = None

# ---- HTML template sans caracteres problematiques ----
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindCheck - Sante Mentale Etudiante</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0a0a12;
            color: #e8e8f0;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #1a1a2c 0%, #0a0a12 100%);
            border-radius: 20px;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #7c5cfc, #fc5c7d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: #888;
            margin-bottom: 20px;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .stat-card {
            background: #1a1a2c;
            padding: 15px 25px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #5cf9c1;
        }

        .stat-label {
            font-size: 0.8rem;
            color: #888;
            margin-top: 5px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }

        .form-card, .result-card, .info-card {
            background: #1a1a2c;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
        }

        .form-title {
            font-size: 1.2rem;
            margin-bottom: 20px;
            color: #7c5cfc;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .form-field {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        label {
            font-size: 0.8rem;
            color: #888;
            text-transform: uppercase;
        }

        select, input {
            background: #12121e;
            border: 1px solid #2a2a3e;
            border-radius: 8px;
            padding: 10px;
            color: white;
            font-size: 0.9rem;
        }

        select:focus, input:focus {
            outline: none;
            border-color: #7c5cfc;
        }

        .toggle-group {
            display: flex;
            gap: 10px;
        }

        .toggle-btn {
            flex: 1;
            padding: 8px;
            border: 1px solid #2a2a3e;
            border-radius: 8px;
            background: #12121e;
            cursor: pointer;
            text-align: center;
            transition: all 0.3s;
        }

        .toggle-btn.active-yes {
            background: #e74c3c;
            border-color: #e74c3c;
            color: white;
        }

        .toggle-btn.active-no {
            background: #2ecc71;
            border-color: #2ecc71;
            color: white;
        }

        .predict-btn {
            width: 100%;
            padding: 14px;
            margin-top: 20px;
            background: linear-gradient(135deg, #7c5cfc, #fc5c7d);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
        }

        .predict-btn:active {
            transform: translateY(0);
        }

        .result-emoji {
            font-size: 3rem;
            text-align: center;
            margin-bottom: 15px;
        }

        .result-level {
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }

        .result-advice {
            text-align: center;
            color: #888;
            line-height: 1.5;
            margin: 15px 0;
        }

        .confidence-bar {
            background: #12121e;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 15px 0;
        }

        .confidence-fill {
            height: 100%;
            background: #5cf9c1;
            transition: width 0.5s;
        }

        .confidence-text {
            text-align: center;
            font-size: 0.8rem;
            color: #888;
        }

        .status-ok {
            color: #2ecc71;
        }

        .status-bad {
            color: #e74c3c;
        }

        .history-list {
            margin-top: 15px;
        }

        .history-item {
            background: #12121e;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }

        .loading {
            display: none;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .hidden {
            display: none;
        }

        footer {
            text-align: center;
            padding: 30px;
            color: #555;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Student MindCheck</h1>
            <div class="subtitle">Detection du risque d anxiete chez les etudiants</div>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="stat-samples">{{ metrics.get('n_samples', '-') }}</div>
                    <div class="stat-label">Etudiants analyses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-prevalence">{{ metrics.get('anxiety_prevalence', '-') }}%</div>
                    <div class="stat-label">Prevalence anxiete</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-model">{{ metrics.get('best_model', '-') }}</div>
                    <div class="stat-label">Meilleur modele</div>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="form-card">
                <div class="form-title">Votre profil etudiant</div>
                <div class="form-grid">
                    <div class="form-field">
                        <label>Genre</label>
                        <select id="gender">
                            <option value="0">Feminin</option>
                            <option value="1">Masculin</option>
                        </select>
                    </div>

                    <div class="form-field">
                        <label>Age</label>
                        <input type="number" id="age" value="20" min="16" max="35">
                    </div>

                    <div class="form-field">
                        <label>Annee d etude</label>
                        <select id="year">
                            <option value="1">1ere annee</option>
                            <option value="2">2eme annee</option>
                            <option value="3" selected>3eme annee</option>
                            <option value="4">4eme annee</option>
                        </select>
                    </div>

                    <div class="form-field">
                        <label>CGPA</label>
                        <select id="cgpa">
                            <option value="0">0 - 1.99</option>
                            <option value="1">2.00 - 2.49</option>
                            <option value="2">2.50 - 2.99</option>
                            <option value="3" selected>3.00 - 3.49</option>
                            <option value="4">3.50 - 4.00</option>
                        </select>
                    </div>

                    <div class="form-field">
                        <label>Marie(e)</label>
                        <div class="toggle-group" id="married-group">
                            <div class="toggle-btn active-no" data-val="0">Non</div>
                            <div class="toggle-btn" data-val="1">Oui</div>
                        </div>
                        <input type="hidden" id="married" value="0">
                    </div>

                    <div class="form-field">
                        <label>Depression</label>
                        <div class="toggle-group" id="depression-group">
                            <div class="toggle-btn active-no" data-val="0">Non</div>
                            <div class="toggle-btn" data-val="1">Oui</div>
                        </div>
                        <input type="hidden" id="depression" value="0">
                    </div>

                    <div class="form-field">
                        <label>Attaques de panique</label>
                        <div class="toggle-group" id="panic-group">
                            <div class="toggle-btn active-no" data-val="0">Non</div>
                            <div class="toggle-btn" data-val="1">Oui</div>
                        </div>
                        <input type="hidden" id="panic_attack" value="0">
                    </div>

                    <div class="form-field">
                        <label>Traitement</label>
                        <div class="toggle-group" id="treatment-group">
                            <div class="toggle-btn active-no" data-val="0">Non</div>
                            <div class="toggle-btn" data-val="1">Oui</div>
                        </div>
                        <input type="hidden" id="treatment" value="0">
                    </div>
                </div>

                <button class="predict-btn" id="predictBtn">
                    <span id="btnText">Analyser mon profil</span>
                    <span id="btnLoader" class="loading"><span class="spinner"></span></span>
                </button>
            </div>

            <div>
                <div class="result-card hidden" id="resultCard">
                    <div class="result-emoji" id="resultEmoji">🤔</div>
                    <div class="result-level" id="resultLevel">En attente...</div>
                    <div class="result-advice" id="resultAdvice">Remplissez le formulaire et cliquez sur Analyser</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidenceBar" style="width: 0%"></div>
                    </div>
                    <div class="confidence-text" id="confidenceText">Confiance : -</div>
                </div>

                <div class="info-card">
                    <div class="form-title">Performance des modeles</div>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #2a2a3e;">
                            <th style="text-align: left; padding: 8px 0;">Modele</th>
                            <th style="text-align: right; padding: 8px 0;">Accuracy</th>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0;">Regression Logistique</td>
                            <td style="text-align: right; padding: 8px 0; color: #2ecc71;">{{ metrics.get('logistic_regression', {}).get('accuracy', '-') }}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0;">Random Forest</td>
                            <td style="text-align: right; padding: 8px 0; color: #5cf9c1;">{{ metrics.get('random_forest', {}).get('accuracy', '-') }}%</td>
                        </tr>
                    </table>
                </div>

                <div class="info-card">
                    <div class="form-title">Historique</div>
                    <div class="history-list" id="historyList">
                        <div style="color: #888; text-align: center;">Aucune prediction</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Projet ML - 4 ING GL - Dr. N. OUERHANI - 2025-2026
        </footer>
    </div>

    <script>
        // Initialisation des boutons toggle
        function initToggles() {
            const toggleGroups = ['married', 'depression', 'panic', 'treatment'];
            
            toggleGroups.forEach(groupName => {
                const group = document.getElementById(groupName + '-group');
                if (!group) return;
                
                const buttons = group.querySelectorAll('.toggle-btn');
                const hiddenInput = document.getElementById(groupName === 'panic' ? 'panic_attack' : groupName);
                
                buttons.forEach(btn => {
                    btn.addEventListener('click', function() {
                        const value = parseInt(this.dataset.val);
                        
                        // Update hidden input
                        if (hiddenInput) hiddenInput.value = value;
                        
                        // Update active classes
                        buttons.forEach(b => {
                            b.classList.remove('active-yes', 'active-no');
                        });
                        
                        if (value === 1) {
                            this.classList.add('active-yes');
                        } else {
                            this.classList.add('active-no');
                        }
                    });
                });
            });
        }
        
        // Historique des predictions
        let history = [];
        
        // Afficher le resultat
        function displayResult(data) {
            const resultCard = document.getElementById('resultCard');
            resultCard.classList.remove('hidden');
            
            const prediction = data.prediction;
            const confidence = Math.round(data.confidence * 100);
            
            const emoji = document.getElementById('resultEmoji');
            const level = document.getElementById('resultLevel');
            const advice = document.getElementById('resultAdvice');
            const bar = document.getElementById('confidenceBar');
            const confText = document.getElementById('confidenceText');
            
            if (prediction === 1) {
                emoji.innerHTML = '😰';
                level.innerHTML = 'Anxiete detectee';
                level.className = 'result-level status-bad';
                advice.innerHTML = '⚠️ Le modele suggere la presence d anxiete. Consultez un professionnel de sante.';
            } else {
                emoji.innerHTML = '😊';
                level.innerHTML = 'Pas d anxiete detectee';
                level.className = 'result-level status-ok';
                advice.innerHTML = '✅ Le modele ne detecte pas de signe d anxiete. Continuez a prendre soin de vous.';
            }
            
            bar.style.width = confidence + '%';
            confText.innerHTML = 'Confiance : ' + confidence + '%';
        }
        
        // Ajouter a l'historique
        function addToHistory(data) {
            const label = data.prediction === 1 ? '⚠️ Anxieux' : '✅ Non anxieux';
            const conf = Math.round(data.confidence * 100);
            
            history.unshift({label: label, conf: conf + '%'});
            if (history.length > 5) history.pop();
            
            const historyDiv = document.getElementById('historyList');
            if (history.length === 0) {
                historyDiv.innerHTML = '<div style="color: #888; text-align: center;">Aucune prediction</div>';
            } else {
                historyDiv.innerHTML = history.map(h => 
                    '<div class="history-item"><span>' + h.label + '</span><span>' + h.conf + '</span></div>'
                ).join('');
            }
        }
        
        // Faire la prediction
        async function makePrediction() {
            const btnText = document.getElementById('btnText');
            const btnLoader = document.getElementById('btnLoader');
            
            // Collecter les donnees
            const data = {
                gender: parseInt(document.getElementById('gender').value),
                age: parseInt(document.getElementById('age').value),
                course: 0,
                year: parseInt(document.getElementById('year').value),
                cgpa: parseInt(document.getElementById('cgpa').value),
                married: parseInt(document.getElementById('married').value),
                depression: parseInt(document.getElementById('depression').value),
                panic_attack: parseInt(document.getElementById('panic_attack').value),
                treatment: parseInt(document.getElementById('treatment').value)
            };
            
            // Afficher le loader
            btnText.style.display = 'none';
            btnLoader.style.display = 'inline-block';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.error) {
                    alert('Erreur: ' + result.error);
                } else {
                    displayResult(result);
                    addToHistory(result);
                }
            } catch (error) {
                console.error('Erreur:', error);
                alert('Erreur de connexion au serveur');
            } finally {
                btnText.style.display = 'inline';
                btnLoader.style.display = 'none';
            }
        }
        
        // Initialisation au chargement
        document.addEventListener('DOMContentLoaded', function() {
            initToggles();
            
            const predictBtn = document.getElementById('predictBtn');
            if (predictBtn) {
                predictBtn.addEventListener('click', makePrediction);
            }
        });
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    if METRICS:
        m = METRICS
    else:
        m = {
            "n_samples": "N/A",
            "anxiety_prevalence": "N/A",
            "best_model": "N/A",
            "logistic_regression": {"accuracy": "N/A", "cv": "N/A"},
            "random_forest": {"accuracy": "N/A", "cv": "N/A"}
        }
    return render_template_string(HTML_PAGE, metrics=m)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modele non charge"}), 500

    try:
        data = request.get_json()
        print("Prediction request:", data)

        feature_vector = [data[f] for f in FEATURE_NAMES]
        X = scaler.transform([feature_vector])
        
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        conf = float(max(proba))
        
        print(f"Prediction: {pred}, Confidence: {conf}")

        # Sauvegarde historique
        try:
            conn = sqlite3.connect('predictions_history.db')
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS history
                           (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                            features TEXT, 
                            prediction INTEGER, 
                            confidence REAL)''')
            cur.execute("INSERT INTO history (features, prediction, confidence) VALUES (?,?,?)",
                       (json.dumps(data), pred, conf))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB error: {e}")

        return jsonify({"prediction": pred, "confidence": conf})

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Application demarree sur http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)