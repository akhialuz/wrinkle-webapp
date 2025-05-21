from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv

app = Flask(__name__)
model = load_model('mobilenet_wrinkle_model.h5')

class_names = ['mild_wrinkle', 'moderate_wrinkle', 'no_wrinkle', 'severe_wrinkle']
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

recommendations = {}
with open('skincare_recommendations.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = f"{row['gender']}_{row['age']}_{row['skin_type']}_{row['wrinkle']}"
        recommendations[key] = row['recommendation']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scan-method')
def scan_method():
    return render_template('method_selection_page.html')

@app.route('/qr-instructions')
def qr_instructions():
    return render_template('qr_instructions.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    prediction = None
    recommendation = None
    uploaded_image = None
    gender = None
    age = None
    skin_type = None

    if request.method == 'POST':
        gender = request.form.get('gender')
        age = request.form.get('age')
        skin_type = request.form.get('skin_type')
        file = request.files.get('file')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            uploaded_image = filepath

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            result = model.predict(img_array)
            predicted_class = class_names[np.argmax(result)]
            readable_label = predicted_class.replace('_', ' ').title()
            prediction = f"Predicted Wrinkle Severity: {readable_label}"

            if predicted_class == "no_wrinkle":
                recommendation = (
                    "Your skin looks amazing! 🌟 Keep up the good work and don’t forget to wear sunscreen every day ☀️"
                )
            else:
                key = f"{gender}_{age}_{skin_type}_{predicted_class}"
                recommendation = recommendations.get(
                    key,
                    "No specific routine found for this combination, but keep taking care of your skin! 💧"
                )

            return render_template('result.html',
                                   gender=gender.title(),
                                   age=age,
                                   skin_type=skin_type.title(),
                                   prediction=prediction,
                                   uploaded_image=uploaded_image,
                                   recommendation=recommendation)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
