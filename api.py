from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

api = Flask(__name__)
CORS(api)

MODEL_PATH = 'saved_model.pkl'  
ENCODERS_PATH = 'encoders.pkl'

def train_and_save():
    df = pd.read_csv('car_evaluation.csv', header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)

    print("Model and encoders saved.")
    return model, encoders

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    print("Loaded existing model and encoders.")
else:
    print("Training model from scratch...")
    model, encoders = train_and_save()


@api.route('/api/car/prediction', methods=['POST'])
def predict_car():
    try:
        data = request.get_json()['inputs']
        
        cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        input_df = pd.DataFrame([data], columns=cols)

        for col in input_df.columns:
            val_str = str(input_df.loc[0, col]).strip()
            encoded_value = -1
            for i, class_val in enumerate(encoders[col].classes_):
                if str(class_val).strip() == val_str:
                    encoded_value = i
                    break
            if encoded_value == -1:
                return jsonify({
                    'error': f"Invalid value '{val_str}' for field '{col}'. "
                             f"Valid values: {list(encoders[col].classes_)}"
                }), 400
            input_df[col] = encoded_value

        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[0]
        pred_label = encoders['class'].inverse_transform(prediction)[0]

        class_labels = encoders['class'].classes_.tolist()
        prob_dict = {label: round(float(prob) * 100, 1)
                     for label, prob in zip(class_labels, probabilities)}

        return jsonify({
            'prediction': pred_label,
            'probabilities': prob_dict
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/api/car/options', methods=['GET'])
def get_options():
    options = {}
    feature_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    for col in feature_cols:
        options[col] = encoders[col].classes_.tolist()
    return jsonify(options)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    api.run(host='0.0.0.0', port=port, debug=False)
