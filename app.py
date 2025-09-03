import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# إنشاء التطبيق
app = Flask(__name__)

# تحميل النموذج و scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('home.html')

# API endpoint لتنبؤات JSON
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)[0]
    return jsonify({'prediction': float(output)})

# Form submission لتنبؤات من HTML
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The House price prediction is {output}")

# تشغيل التطبيق على Render أو محلي
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render يحدد PORT تلقائيًا
    app.run(host="0.0.0.0", port=port, debug=True)
