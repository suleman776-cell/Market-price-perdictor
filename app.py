from flask import Flask, render_template, request
import pickle
import pandas as pd
import traceback
from datetime import datetime
import os

app = Flask(__name__)

# --- Model Load ---
# Ensure your 'ecommerce_model.pkl' is in the same directory
try:
    model_path = "ecommerce_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Model Load Error: {e}. Make sure ecommerce_model.pkl exists.")
    model = None

# --- Logical Mappings ---
MAPS = {
    'category_list': ['Electronics', 'Fashion', 'Accessories', 'Computers', 'Wearables'],
    'brands_by_category': {
        'Electronics': ['Samsung', 'Apple', 'Xiaomi', 'JBL'],
        'Fashion': ['Nike', 'Adidas', 'Generic'],
        'Accessories': ['JBL', 'Generic'],
        'Computers': ['HP', 'Dell', 'Apple'],
        'Wearables': ['FitPro', 'Apple', 'Samsung']
    },
    'category_enc': {'Electronics': 0, 'Fashion': 1, 'Accessories': 2, 'Computers': 3, 'Wearables': 4},
    'brand_enc': {
        'Samsung': 0, 'Nike': 1, 'JBL': 2, 'HP': 3, 'Apple': 4, 
        'Generic': 5, 'FitPro': 6, 'Xiaomi': 7, 'Dell': 8, 'Adidas': 9
    },
    'platform_enc': {'Souq': 0, 'Jumia': 1, 'Amazon': 2},
    'city_enc': {'Cairo': 0, 'Alexandria': 1, 'Casablanca': 2, 'Dubai': 3, 'Riyadh': 4, 'Giza': 5}
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
        if model is None:
            return render_template('index.html', error="Model file not found!", maps=MAPS)

        def safe_float(value, default=0.0):
            try:
                return float(value) if value and str(value).strip() else default
            except:
                return default

        category_selected = request.form.get('category')
        brand_selected = request.form.get('brand')
        city_selected = request.form.get('city')
        qty = safe_float(request.form.get('quantity'), 1.0)
        
        now = datetime.now()

        data_dict = {
            'Product': 0, 
            'Category': MAPS['category_enc'].get(category_selected, 0),
            'Brand': MAPS['brand_enc'].get(brand_selected, 5),
            'Platform': MAPS['platform_enc'].get(request.form.get('platform'), 0),
            'City': MAPS['city_enc'].get(city_selected, 0),
            'Quantity': qty,
            'TotalAmount': 0, 
            'Rating': safe_float(request.form.get('rating'), 4.0),
            'Reviews': safe_float(request.form.get('reviews'), 100.0),
            'Day': now.day,
            'Month': now.month,
            'year': now.year
        }

        input_df = pd.DataFrame([data_dict])

        # Feature Order handling
        try:
            expected_order = model.feature_names_in_
            input_df = input_df[expected_order]
        except AttributeError:
            expected_order = ['Product', 'Category', 'Brand', 'Platform', 'City', 
                             'Quantity', 'TotalAmount', 'Rating', 'Reviews', 'Day', 'Month', 'year']
            input_df = input_df[expected_order]

        prediction_usd = model.predict(input_df)[0]

        # Currency Conversion logic
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