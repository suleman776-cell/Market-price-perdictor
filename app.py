from flask import Flask, render_template, request
import pickle
import pandas as pd
import traceback
from datetime import datetime

app = Flask(__name__)

# Model Load
try:
    with open("ecommerce_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Model Load Error: {e}")

# Dataset se nikaale gaye complete mappings
MAPS = {
    'category': {'Electronics': 0, 'Fashion': 1, 'Accessories': 2, 'Computers': 3, 'Wearables': 4},
    'brand': {'Samsung': 0, 'Nike': 1, 'JBL': 2, 'HP': 3, 'Apple': 4, 'Generic': 5, 'FitPro': 6, 'Xiaomi': 7, 'Dell': 8, 'Adidas': 9},
    'platform': {'Souq': 0, 'Jumia': 1, 'Amazon': 2},
    'city': {'Cairo': 0, 'Alexandria': 1, 'Casablanca': 2, 'Dubai': 3, 'Riyadh': 4, 'Giza': 5}
}

CURRENCY_CONFIG = {
    'Dubai': {'symbol': 'AED', 'rate': 3.67},
    'Cairo': {'symbol': 'EGP', 'rate': 48.50},
    'Alexandria': {'symbol': 'EGP', 'rate': 48.50},
    'Giza': {'symbol': 'EGP', 'rate': 48.50},
    'Casablanca': {'symbol': 'MAD', 'rate': 10.10},
    'Riyadh': {'symbol': 'SAR', 'rate': 3.75}
}

@app.route('/')
def home():
    return render_template('index.html', maps=MAPS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        def safe_float(value, default=0.0):
            try:
                return float(value) if value and str(value).strip() else default
            except:
                return default

        city_selected = request.form.get('city')
        now = datetime.now()
        qty = safe_float(request.form.get('quantity'), 1.0)

        # 1. Input Data taiyar karein
        data_dict = {
            'Product': 0, # Product names usually encoding mangte hain, agar training mein tha
            'Category': MAPS['category'].get(request.form.get('category'), 0),
            'Brand': MAPS['brand'].get(request.form.get('brand'), 0),
            'Platform': MAPS['platform'].get(request.form.get('platform'), 0),
            'City': MAPS['city'].get(city_selected, 0),
            'Quantity': qty,
            'TotalAmount': 0, # NOTE: Agar model training mein Price predict kar raha tha, 
                             # toh TotalAmount input feature nahi hona chahiye tha. 
                             # Agar prediction galat aaye toh model retraining mein ise hata dein.
            'Rating': safe_float(request.form.get('rating'), 4.0),
            'Reviews': safe_float(request.form.get('reviews'), 100.0),
            'Day': now.day,
            'Month': now.month,
            'year': now.year
        }

        input_df = pd.DataFrame([data_dict])

        # 2. Sahi Feature Order (Zaroori Step)
        try:
            expected_order = model.feature_names_in_
            input_df = input_df[expected_order]
        except AttributeError:
            # Manually order agar model meta-data nahi de raha
            expected_order = ['Product', 'Category', 'Brand', 'Platform', 'City', 
                             'Quantity', 'TotalAmount', 'Rating', 'Reviews', 'Day', 'Month', 'year']
            input_df = input_df[expected_order]

        # 3. Prediction
        prediction_usd = model.predict(input_df)[0]

        # 4. Currency Conversion
        curr = CURRENCY_CONFIG.get(city_selected, {'symbol': '$', 'rate': 1})
        final_price = float(prediction_usd) * curr['rate']
        formatted_price = f"{curr['symbol']} {final_price:,.2f}"

        return render_template('index.html', 
                               prediction_text=formatted_price, 
                               selected_city=city_selected,
                               maps=MAPS)

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html', error=f"Error: {str(e)}", maps=MAPS)

if __name__ == "__main__":
    app.run(debug=True)