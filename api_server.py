<<<<<<< HEAD
# api_server.py
import pandas as pd
import numpy as np
import joblib
import sklearn # Good practice
from flask import Flask, jsonify
from flask_cors import CORS
import os # To handle file paths reliably

# --- --- Configuration --- ---
# Assuming api_server.py is in the root project directory
# Adjust these paths if your script/data/models are elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'churn_pipeline.joblib')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'processed_feature_names.joblib')
RAW_CUSTOMER_DATA_PATH = os.path.join(BASE_DIR, 'Cleaned_E_Commerce_Data.csv')

# --- Recommendation Thresholds (Keep as they were) ---
HIGH_RISK_THRESHOLD = 0.60
INACTIVITY_DAYS_THRESHOLD = 30
LOW_SATISFACTION_THRESHOLD = 2
LOW_TENURE_MONTHS_THRESHOLD = 6
NEGATIVE_ORDER_HIKE_THRESHOLD = -5
LOW_APP_HOURS_THRESHOLD = 1.5
HIGH_CHURN_PAYMENT_MODES = ['Debit Card']
RETENTION_PAYMENT_MODES = ['Cash on Delivery', 'CC']
TOP_N_FEATURES_TO_CONSIDER = 5
TOP_N_RECOMMENDATIONS = 50 # How many rows to return

# --- --- Global Variables for Loaded Components --- ---
loaded_pipeline = None
feature_names_processed = None
customer_data_raw = None
fitted_preprocessor = None
trained_model = None
feature_importances = None
important_feature_names = [] # Initialize globally

# --- --- Loading Function (to run once on startup) --- ---
def load_components():
    global loaded_pipeline, feature_names_processed, customer_data_raw
    global fitted_preprocessor, trained_model, feature_importances, important_feature_names
    try:
        print(f"API: Loading pipeline from: {PIPELINE_PATH}")
        loaded_pipeline = joblib.load(PIPELINE_PATH)
        print("API: Pipeline loaded.")

        print(f"API: Loading feature names from: {FEATURE_NAMES_PATH}")
        feature_names_processed = joblib.load(FEATURE_NAMES_PATH)
        print("API: Feature names loaded.")

        print(f"API: Loading raw customer data from: {RAW_CUSTOMER_DATA_PATH}")
        customer_data_raw = pd.read_csv(RAW_CUSTOMER_DATA_PATH)
        if 'CustomerID' not in customer_data_raw.columns:
             raise ValueError("Raw data must contain 'CustomerID'.")
        # Ensure index is standard if it's not already
        # customer_data_raw = customer_data_raw.reset_index(drop=True)
        print(f"API: Raw customer data loaded. Shape: {customer_data_raw.shape}")


        # Extract components from pipeline
        fitted_preprocessor = loaded_pipeline.named_steps['preprocessor']
        trained_model = loaded_pipeline.named_steps['model']
        print("API: Preprocessor and model extracted.")

        # Extract feature importances
        if hasattr(trained_model, 'feature_importances_'):
            feature_importances = trained_model.feature_importances_
            if len(feature_importances) == len(feature_names_processed):
                indices = np.argsort(feature_importances)[::-1]
                important_feature_names = [feature_names_processed[i] for i in indices[:TOP_N_FEATURES_TO_CONSIDER]]
                print(f"API: Top {TOP_N_FEATURES_TO_CONSIDER} features identified.")
            else:
                 print("API Warning: Feature importance/names length mismatch.")
                 feature_importances = None # Reset if mismatch
        else:
            print("API: Feature importances not available for this model.")

        print("--- API: Components loaded successfully ---")
        return True

    except FileNotFoundError as e:
        print(f"API Error: File not found during loading. Make sure '{e.filename}' exists.")
        return False
    except KeyError as e:
         print(f"API Error: Pipeline step '{e}' not found. Steps: {list(loaded_pipeline.named_steps.keys())}")
         return False
    except Exception as e:
        print(f"API Error: An unexpected error occurred during loading: {e}")
        return False

# --- --- Recommendation Generation Logic (modified slightly) --- ---
def generate_recommendations_for_api(
    customer_data: pd.DataFrame, # Use loaded data
    model,
    preprocessor,
    f_names_processed: list,
    f_importances: np.ndarray,
    imp_feature_names: list # Use loaded important names
    ):
    """Generates recommendations and returns a list of dictionaries."""
    recommendations_dict = {} # Renamed inner variable
    results_list = [] # List to store final row dicts

    if customer_data is None or model is None or preprocessor is None:
         print("API Error: Components not loaded correctly.")
         return [] # Return empty list on error

    if 'CustomerID' not in customer_data.columns:
        raise ValueError("customer_data must include a 'CustomerID' column.")

    customer_ids = customer_data['CustomerID']
    X_raw = customer_data.drop('CustomerID', axis=1, errors='ignore')

    try:
        X_processed = preprocessor.transform(X_raw)
    except Exception as e:
        print(f"API Error during preprocessing: {e}")
        return []

    try:
        churn_probabilities = model.predict_proba(X_processed)[:, 1]
    except Exception as e:
        print(f"API Error during prediction: {e}")
        return []


    churn_predictions_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'ChurnProbability': churn_probabilities,
        'OriginalIndex': customer_ids.index
    })

    high_risk_df = churn_predictions_df[churn_predictions_df['ChurnProbability'] >= HIGH_RISK_THRESHOLD].copy()
    high_risk_df.sort_values(by='ChurnProbability', ascending=False, inplace=True)
    print(f"API: Found {len(high_risk_df)} high-risk customers.")

    # Limit to TOP_N_RECOMMENDATIONS customers here if needed,
    # but easier to limit the final output rows later
    # high_risk_df = high_risk_df.head(SOME_CUSTOMER_LIMIT)


    for _, high_risk_row in high_risk_df.iterrows():
        customer_id = high_risk_row['CustomerID']
        original_idx = high_risk_row['OriginalIndex']
        # Use .loc with the original index label
        try:
            customer_record = customer_data.loc[original_idx]
        except KeyError:
            print(f"API Warning: Could not find original index {original_idx} in raw data. Skipping customer {customer_id}.")
            continue # Skip if index somehow became invalid

        customer_recs_list = []

        # --- Apply Rule-Based Logic (Copied from previous script)---

        # 1. Based on DaySinceLastOrder
        if 'DaySinceLastOrder' in customer_record and pd.notna(customer_record['DaySinceLastOrder']) and customer_record['DaySinceLastOrder'] > INACTIVITY_DAYS_THRESHOLD:
            cause = f"High Inactivity ({int(customer_record['DaySinceLastOrder'])} days)"
            rec = f"Re-engagement Campaign: Offer special discount/reminder."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 2. Based on SatisfactionScore
        if 'SatisfactionScore' in customer_record and pd.notna(customer_record['SatisfactionScore']) and customer_record['SatisfactionScore'] <= LOW_SATISFACTION_THRESHOLD:
            cause = f"Low Satisfaction Score ({int(customer_record['SatisfactionScore'])})"
            rec = "Proactive support outreach, feedback survey, or goodwill gesture needed."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 3. Based on Tenure
        if 'Tenure' in customer_record and pd.notna(customer_record['Tenure']) and customer_record['Tenure'] <= LOW_TENURE_MONTHS_THRESHOLD:
            cause = f"Low Tenure ({int(customer_record['Tenure'])} months)"
            rec = "Focus on onboarding, tutorials, first-purchase offers, or welcome check-in."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 4. Based on CashbackAmount / CouponUsed Importance (using globally loaded imp_feature_names)
        if 'CashbackAmount' in imp_feature_names or 'CouponUsed' in imp_feature_names:
            cause = "High Sensitivity to Incentives (Cashback/Coupon)"
            rec = "Check usage of cashback/coupons. Consider targeted loyalty offer."
            if not any(d['Cause'].startswith('Review Incentives') for d in customer_recs_list):
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
        elif 'CashbackAmount' in customer_record and pd.notna(customer_record['CashbackAmount']):
            cause = f"Review Incentives (Cashback: {customer_record['CashbackAmount']:.2f})"
            rec = "Consider personalized coupon/cashback if activity is low or decreasing."
            if not any(d['Cause'].startswith('High Sensitivity to Incentives') for d in customer_recs_list):
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 5. Based on Payment Mode
        if 'PreferredPaymentMode' in customer_record and pd.notna(customer_record['PreferredPaymentMode']):
            pm = customer_record['PreferredPaymentMode']
            if pm in HIGH_CHURN_PAYMENT_MODES:
                cause = f"Using High-Risk Payment Mode ({pm})"
                rec = f"Investigate friction. Consider incentive to switch to modes like {RETENTION_PAYMENT_MODES}."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
            elif pm not in RETENTION_PAYMENT_MODES:
                cause = f"Not Using Preferred Payment Mode ({pm})"
                rec = f"Encourage trial of preferred modes ({RETENTION_PAYMENT_MODES}) for potentially better retention."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 6. Based on Personalized Outreach (Order Category)
        if 'PreferedOrderCat' in customer_record and pd.notna(customer_record['PreferedOrderCat']):
            fav_cat = customer_record['PreferedOrderCat']
            cause = f"Opportunity: Preferred Category ('{fav_cat}')"
            rec = f"Target with promotion related to '{fav_cat}'. E.g., '10% off your next {fav_cat} order!'"
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 7. Based on OrderAmountHikeFromlastYear
        if 'OrderAmountHikeFromlastYear' in customer_record and pd.notna(customer_record['OrderAmountHikeFromlastYear']):
            hike = customer_record['OrderAmountHikeFromlastYear']
            if hike <= NEGATIVE_ORDER_HIKE_THRESHOLD:
                cause = f"Decreased Spending ({hike:.1f}%)"
                rec = "Target with value bundles or special promotions."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
            elif -5 < hike < 5:
                cause = "Stagnant Spending"
                rec = "Introduce relevant new products or categories based on purchase history."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 8. Based on App/Website Engagement
        if 'HourSpendOnApp' in customer_record and pd.notna(customer_record['HourSpendOnApp']) and customer_record['HourSpendOnApp'] < LOW_APP_HOURS_THRESHOLD:
            cause = f"Low App Engagement ({customer_record['HourSpendOnApp']:.1f} hrs)"
            rec = "Investigate app usability/content. Use targeted in-app messages for features/offers."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # --- --- ---

        # If recommendations were generated, add rows to results_list
        if customer_recs_list:
            for rec_dict in customer_recs_list:
                results_list.append({
                    'CustomerID': customer_id,
                    'Cause': rec_dict['Cause'], # Renamed for consistency
                    'Recommendation': rec_dict['Recommendation']
                })
        else:
             # Add the generic message if no specific rules triggered for this high-risk user
             results_list.append({
                    'CustomerID': customer_id,
                    'Cause': 'High Churn Risk (Generic)',
                    'Recommendation': "No specific rule triggered. Review manually or broaden campaign."
             })

        # Stop adding rows if we already have enough for the top N output
        if len(results_list) >= TOP_N_RECOMMENDATIONS:
            break # Exit the customer loop once we have enough rows

    print(f"API: Generated {len(results_list)} recommendation rows.")
    # Return only the top N rows
    return results_list[:TOP_N_RECOMMENDATIONS]


# --- --- Flask App Setup --- ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for frontend requests

# --- --- Load components when Flask starts --- ---
components_loaded = load_components()

# --- --- API Endpoint --- ---
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    if not components_loaded:
         # Return error in a consistent JSON format
         return jsonify({"error": "API components failed to load on startup. Check server logs.", "threshold": None, "recommendations": []}), 500

    print("API: Received request for /api/recommendations")
    try:
        recommendation_data = generate_recommendations_for_api(
            customer_data=customer_data_raw,
            model=trained_model,
            preprocessor=fitted_preprocessor,
            f_names_processed=feature_names_processed,
            f_importances=feature_importances,
            imp_feature_names=important_feature_names
        )

        if recommendation_data is None: # Check if generation failed internally
             # Return error in a consistent JSON format
             return jsonify({"error": "Recommendation generation failed. Check server logs.", "threshold": HIGH_RISK_THRESHOLD, "recommendations": []}), 500

        print(f"API: Sending {len(recommendation_data)} recommendation rows.")

        # --- Create the response payload with threshold ---
        response_payload = {
            "threshold": HIGH_RISK_THRESHOLD, # Add the threshold value here
            "recommendations": recommendation_data
        }
        # --- --- --- --- --- --- --- --- --- --- --- --- ---

        return jsonify(response_payload) # Return the combined payload

    except Exception as e:
        print(f"API Error during request processing: {e}")
        # Return error in a consistent JSON format
        return jsonify({"error": f"An unexpected error occurred on the server: {e}", "threshold": HIGH_RISK_THRESHOLD, "recommendations": []}), 500


    except Exception as e:
        print(f"API Error during request processing: {e}")
        # Optionally log the full traceback here
        return jsonify({"error": f"An unexpected error occurred on the server: {e}"}), 500


# --- --- NEW: Graph Data API Endpoints --- ---

@app.route('/api/graph_data/revenue_pie', methods=['GET'])
def get_revenue_pie_data():
    if not components_loaded or customer_data_raw is None:
        return jsonify({"error": "API components not ready."}), 500
    if 'Churn' not in customer_data_raw.columns or 'OrderAmountHikeFromlastYear' not in customer_data_raw.columns:
         return jsonify({"error": "Required columns (Churn, OrderAmountHikeFromlastYear) not found in loaded data."}), 500

    try:
        # Ensure 'Churn' column exists and is numeric (0 or 1)
        if not pd.api.types.is_numeric_dtype(customer_data_raw['Churn']):
             # Attempt conversion if possible (e.g., if it's 'Yes'/'No')
             # This is just an example, adjust based on your actual data
             # customer_data_raw['Churn'] = customer_data_raw['Churn'].map({'Yes': 1, 'No': 0})
             # If conversion isn't straightforward, return error
             return jsonify({"error": "'Churn' column is not numeric (0/1)."}), 400


        # Calculate total revenue hike for churned (1) and retained (0) customers
        # Use fillna(0) in case there are NaN values in the hike column
        revenue_churned = customer_data_raw[customer_data_raw['Churn'] == 1]['OrderAmountHikeFromlastYear'].fillna(0).sum()
        revenue_retained = customer_data_raw[customer_data_raw['Churn'] == 0]['OrderAmountHikeFromlastYear'].fillna(0).sum()

        # Prepare data for frontend (Recharts PieChart usually expects an array of objects)
        # Handle potential negative values if hike can be negative - Pie charts work best with non-negative values.
        # We'll send raw sums, frontend can decide how to display (maybe absolute values or different chart type if sums negative).
        pie_data = [
            {"name": "Revenue Hike Lost (Churned)", "value": revenue_churned},
            {"name": "Revenue Hike Retained", "value": revenue_retained},
        ]
        # Filter out entries where value might be zero or negative if needed for pie chart logic
        # pie_data = [d for d in pie_data if d['value'] > 0]

        return jsonify(pie_data)

    except Exception as e:
        print(f"API Error in /revenue_pie: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/api/graph_data/payment_distribution', methods=['GET'])
def get_payment_distribution_data():
    if not components_loaded or customer_data_raw is None:
        return jsonify({"error": "API components not ready."}), 500
    if 'Churn' not in customer_data_raw.columns or 'PreferredPaymentMode' not in customer_data_raw.columns:
         return jsonify({"error": "Required columns (Churn, PreferredPaymentMode) not found in loaded data."}), 500

    try:
        # Ensure 'Churn' column exists and is numeric (0 or 1)
        if not pd.api.types.is_numeric_dtype(customer_data_raw['Churn']):
             return jsonify({"error": "'Churn' column is not numeric (0/1)."}), 400

        # Calculate counts using crosstab
        # fillna(0) handles cases where a payment mode might only have churned or non-churned users
        counts = pd.crosstab(customer_data_raw['PreferredPaymentMode'], customer_data_raw['Churn']).fillna(0)

        # Rename columns for clarity (assuming 0=No Churn, 1=Churn)
        counts.rename(columns={0: 'noChurn', 1: 'churn'}, inplace=True)

        # Reset index to make 'PreferredPaymentMode' a column
        counts = counts.reset_index()

        # Convert to list of dictionaries suitable for Recharts BarChart
        # Example format: [{'PreferredPaymentMode': 'Card', 'noChurn': 50, 'churn': 10}, ...]
        chart_data = counts.to_dict(orient='records')

        return jsonify(chart_data)

    except Exception as e:
        print(f"API Error in /payment_distribution: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


# --- --- Run the App --- ---
if __name__ == '__main__':
    # Make sure the host is accessible from your frontend (0.0.0.0)
    # Use a port that doesn't conflict (e.g., 5001 if 5000 is common)
=======
# api_server.py
import pandas as pd
import numpy as np
import joblib
import sklearn # Good practice
from flask import Flask, jsonify
from flask_cors import CORS
import os # To handle file paths reliably

# --- --- Configuration --- ---
# Assuming api_server.py is in the root project directory
# Adjust these paths if your script/data/models are elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'churn_pipeline.joblib')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'processed_feature_names.joblib')
RAW_CUSTOMER_DATA_PATH = os.path.join(BASE_DIR, 'Cleaned_E_Commerce_Data.csv')

# --- Recommendation Thresholds (Keep as they were) ---
HIGH_RISK_THRESHOLD = 0.60
INACTIVITY_DAYS_THRESHOLD = 30
LOW_SATISFACTION_THRESHOLD = 2
LOW_TENURE_MONTHS_THRESHOLD = 6
NEGATIVE_ORDER_HIKE_THRESHOLD = -5
LOW_APP_HOURS_THRESHOLD = 1.5
HIGH_CHURN_PAYMENT_MODES = ['Debit Card']
RETENTION_PAYMENT_MODES = ['Cash on Delivery', 'CC']
TOP_N_FEATURES_TO_CONSIDER = 5
TOP_N_RECOMMENDATIONS = 50 # How many rows to return

# --- --- Global Variables for Loaded Components --- ---
loaded_pipeline = None
feature_names_processed = None
customer_data_raw = None
fitted_preprocessor = None
trained_model = None
feature_importances = None
important_feature_names = [] # Initialize globally

# --- --- Loading Function (to run once on startup) --- ---
def load_components():
    global loaded_pipeline, feature_names_processed, customer_data_raw
    global fitted_preprocessor, trained_model, feature_importances, important_feature_names
    try:
        print(f"API: Loading pipeline from: {PIPELINE_PATH}")
        loaded_pipeline = joblib.load(PIPELINE_PATH)
        print("API: Pipeline loaded.")

        print(f"API: Loading feature names from: {FEATURE_NAMES_PATH}")
        feature_names_processed = joblib.load(FEATURE_NAMES_PATH)
        print("API: Feature names loaded.")

        print(f"API: Loading raw customer data from: {RAW_CUSTOMER_DATA_PATH}")
        customer_data_raw = pd.read_csv(RAW_CUSTOMER_DATA_PATH)
        if 'CustomerID' not in customer_data_raw.columns:
             raise ValueError("Raw data must contain 'CustomerID'.")
        # Ensure index is standard if it's not already
        # customer_data_raw = customer_data_raw.reset_index(drop=True)
        print(f"API: Raw customer data loaded. Shape: {customer_data_raw.shape}")


        # Extract components from pipeline
        fitted_preprocessor = loaded_pipeline.named_steps['preprocessor']
        trained_model = loaded_pipeline.named_steps['model']
        print("API: Preprocessor and model extracted.")

        # Extract feature importances
        if hasattr(trained_model, 'feature_importances_'):
            feature_importances = trained_model.feature_importances_
            if len(feature_importances) == len(feature_names_processed):
                indices = np.argsort(feature_importances)[::-1]
                important_feature_names = [feature_names_processed[i] for i in indices[:TOP_N_FEATURES_TO_CONSIDER]]
                print(f"API: Top {TOP_N_FEATURES_TO_CONSIDER} features identified.")
            else:
                 print("API Warning: Feature importance/names length mismatch.")
                 feature_importances = None # Reset if mismatch
        else:
            print("API: Feature importances not available for this model.")

        print("--- API: Components loaded successfully ---")
        return True

    except FileNotFoundError as e:
        print(f"API Error: File not found during loading. Make sure '{e.filename}' exists.")
        return False
    except KeyError as e:
         print(f"API Error: Pipeline step '{e}' not found. Steps: {list(loaded_pipeline.named_steps.keys())}")
         return False
    except Exception as e:
        print(f"API Error: An unexpected error occurred during loading: {e}")
        return False

# --- --- Recommendation Generation Logic (modified slightly) --- ---
def generate_recommendations_for_api(
    customer_data: pd.DataFrame, # Use loaded data
    model,
    preprocessor,
    f_names_processed: list,
    f_importances: np.ndarray,
    imp_feature_names: list # Use loaded important names
    ):
    """Generates recommendations and returns a list of dictionaries."""
    recommendations_dict = {} # Renamed inner variable
    results_list = [] # List to store final row dicts

    if customer_data is None or model is None or preprocessor is None:
         print("API Error: Components not loaded correctly.")
         return [] # Return empty list on error

    if 'CustomerID' not in customer_data.columns:
        raise ValueError("customer_data must include a 'CustomerID' column.")

    customer_ids = customer_data['CustomerID']
    X_raw = customer_data.drop('CustomerID', axis=1, errors='ignore')

    try:
        X_processed = preprocessor.transform(X_raw)
    except Exception as e:
        print(f"API Error during preprocessing: {e}")
        return []

    try:
        churn_probabilities = model.predict_proba(X_processed)[:, 1]
    except Exception as e:
        print(f"API Error during prediction: {e}")
        return []


    churn_predictions_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'ChurnProbability': churn_probabilities,
        'OriginalIndex': customer_ids.index
    })

    high_risk_df = churn_predictions_df[churn_predictions_df['ChurnProbability'] >= HIGH_RISK_THRESHOLD].copy()
    high_risk_df.sort_values(by='ChurnProbability', ascending=False, inplace=True)
    print(f"API: Found {len(high_risk_df)} high-risk customers.")

    # Limit to TOP_N_RECOMMENDATIONS customers here if needed,
    # but easier to limit the final output rows later
    # high_risk_df = high_risk_df.head(SOME_CUSTOMER_LIMIT)


    for _, high_risk_row in high_risk_df.iterrows():
        customer_id = high_risk_row['CustomerID']
        original_idx = high_risk_row['OriginalIndex']
        # Use .loc with the original index label
        try:
            customer_record = customer_data.loc[original_idx]
        except KeyError:
            print(f"API Warning: Could not find original index {original_idx} in raw data. Skipping customer {customer_id}.")
            continue # Skip if index somehow became invalid

        customer_recs_list = []

        # --- Apply Rule-Based Logic (Copied from previous script)---

        # 1. Based on DaySinceLastOrder
        if 'DaySinceLastOrder' in customer_record and pd.notna(customer_record['DaySinceLastOrder']) and customer_record['DaySinceLastOrder'] > INACTIVITY_DAYS_THRESHOLD:
            cause = f"High Inactivity ({int(customer_record['DaySinceLastOrder'])} days)"
            rec = f"Re-engagement Campaign: Offer special discount/reminder."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 2. Based on SatisfactionScore
        if 'SatisfactionScore' in customer_record and pd.notna(customer_record['SatisfactionScore']) and customer_record['SatisfactionScore'] <= LOW_SATISFACTION_THRESHOLD:
            cause = f"Low Satisfaction Score ({int(customer_record['SatisfactionScore'])})"
            rec = "Proactive support outreach, feedback survey, or goodwill gesture needed."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 3. Based on Tenure
        if 'Tenure' in customer_record and pd.notna(customer_record['Tenure']) and customer_record['Tenure'] <= LOW_TENURE_MONTHS_THRESHOLD:
            cause = f"Low Tenure ({int(customer_record['Tenure'])} months)"
            rec = "Focus on onboarding, tutorials, first-purchase offers, or welcome check-in."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 4. Based on CashbackAmount / CouponUsed Importance (using globally loaded imp_feature_names)
        if 'CashbackAmount' in imp_feature_names or 'CouponUsed' in imp_feature_names:
            cause = "High Sensitivity to Incentives (Cashback/Coupon)"
            rec = "Check usage of cashback/coupons. Consider targeted loyalty offer."
            if not any(d['Cause'].startswith('Review Incentives') for d in customer_recs_list):
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
        elif 'CashbackAmount' in customer_record and pd.notna(customer_record['CashbackAmount']):
            cause = f"Review Incentives (Cashback: {customer_record['CashbackAmount']:.2f})"
            rec = "Consider personalized coupon/cashback if activity is low or decreasing."
            if not any(d['Cause'].startswith('High Sensitivity to Incentives') for d in customer_recs_list):
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 5. Based on Payment Mode
        if 'PreferredPaymentMode' in customer_record and pd.notna(customer_record['PreferredPaymentMode']):
            pm = customer_record['PreferredPaymentMode']
            if pm in HIGH_CHURN_PAYMENT_MODES:
                cause = f"Using High-Risk Payment Mode ({pm})"
                rec = f"Investigate friction. Consider incentive to switch to modes like {RETENTION_PAYMENT_MODES}."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
            elif pm not in RETENTION_PAYMENT_MODES:
                cause = f"Not Using Preferred Payment Mode ({pm})"
                rec = f"Encourage trial of preferred modes ({RETENTION_PAYMENT_MODES}) for potentially better retention."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 6. Based on Personalized Outreach (Order Category)
        if 'PreferedOrderCat' in customer_record and pd.notna(customer_record['PreferedOrderCat']):
            fav_cat = customer_record['PreferedOrderCat']
            cause = f"Opportunity: Preferred Category ('{fav_cat}')"
            rec = f"Target with promotion related to '{fav_cat}'. E.g., '10% off your next {fav_cat} order!'"
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 7. Based on OrderAmountHikeFromlastYear
        if 'OrderAmountHikeFromlastYear' in customer_record and pd.notna(customer_record['OrderAmountHikeFromlastYear']):
            hike = customer_record['OrderAmountHikeFromlastYear']
            if hike <= NEGATIVE_ORDER_HIKE_THRESHOLD:
                cause = f"Decreased Spending ({hike:.1f}%)"
                rec = "Target with value bundles or special promotions."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
            elif -5 < hike < 5:
                cause = "Stagnant Spending"
                rec = "Introduce relevant new products or categories based on purchase history."
                customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # 8. Based on App/Website Engagement
        if 'HourSpendOnApp' in customer_record and pd.notna(customer_record['HourSpendOnApp']) and customer_record['HourSpendOnApp'] < LOW_APP_HOURS_THRESHOLD:
            cause = f"Low App Engagement ({customer_record['HourSpendOnApp']:.1f} hrs)"
            rec = "Investigate app usability/content. Use targeted in-app messages for features/offers."
            customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

        # --- --- ---

        # If recommendations were generated, add rows to results_list
        if customer_recs_list:
            for rec_dict in customer_recs_list:
                results_list.append({
                    'CustomerID': customer_id,
                    'Cause': rec_dict['Cause'], # Renamed for consistency
                    'Recommendation': rec_dict['Recommendation']
                })
        else:
             # Add the generic message if no specific rules triggered for this high-risk user
             results_list.append({
                    'CustomerID': customer_id,
                    'Cause': 'High Churn Risk (Generic)',
                    'Recommendation': "No specific rule triggered. Review manually or broaden campaign."
             })

        # Stop adding rows if we already have enough for the top N output
        if len(results_list) >= TOP_N_RECOMMENDATIONS:
            break # Exit the customer loop once we have enough rows

    print(f"API: Generated {len(results_list)} recommendation rows.")
    # Return only the top N rows
    return results_list[:TOP_N_RECOMMENDATIONS]


# --- --- Flask App Setup --- ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for frontend requests

# --- --- Load components when Flask starts --- ---
components_loaded = load_components()

# --- --- API Endpoint --- ---
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    if not components_loaded:
         # Return error in a consistent JSON format
         return jsonify({"error": "API components failed to load on startup. Check server logs.", "threshold": None, "recommendations": []}), 500

    print("API: Received request for /api/recommendations")
    try:
        recommendation_data = generate_recommendations_for_api(
            customer_data=customer_data_raw,
            model=trained_model,
            preprocessor=fitted_preprocessor,
            f_names_processed=feature_names_processed,
            f_importances=feature_importances,
            imp_feature_names=important_feature_names
        )

        if recommendation_data is None: # Check if generation failed internally
             # Return error in a consistent JSON format
             return jsonify({"error": "Recommendation generation failed. Check server logs.", "threshold": HIGH_RISK_THRESHOLD, "recommendations": []}), 500

        print(f"API: Sending {len(recommendation_data)} recommendation rows.")

        # --- Create the response payload with threshold ---
        response_payload = {
            "threshold": HIGH_RISK_THRESHOLD, # Add the threshold value here
            "recommendations": recommendation_data
        }
        # --- --- --- --- --- --- --- --- --- --- --- --- ---

        return jsonify(response_payload) # Return the combined payload

    except Exception as e:
        print(f"API Error during request processing: {e}")
        # Return error in a consistent JSON format
        return jsonify({"error": f"An unexpected error occurred on the server: {e}", "threshold": HIGH_RISK_THRESHOLD, "recommendations": []}), 500


    except Exception as e:
        print(f"API Error during request processing: {e}")
        # Optionally log the full traceback here
        return jsonify({"error": f"An unexpected error occurred on the server: {e}"}), 500


# --- --- NEW: Graph Data API Endpoints --- ---

@app.route('/api/graph_data/revenue_pie', methods=['GET'])
def get_revenue_pie_data():
    if not components_loaded or customer_data_raw is None:
        return jsonify({"error": "API components not ready."}), 500
    if 'Churn' not in customer_data_raw.columns or 'OrderAmountHikeFromlastYear' not in customer_data_raw.columns:
         return jsonify({"error": "Required columns (Churn, OrderAmountHikeFromlastYear) not found in loaded data."}), 500

    try:
        # Ensure 'Churn' column exists and is numeric (0 or 1)
        if not pd.api.types.is_numeric_dtype(customer_data_raw['Churn']):
             # Attempt conversion if possible (e.g., if it's 'Yes'/'No')
             # This is just an example, adjust based on your actual data
             # customer_data_raw['Churn'] = customer_data_raw['Churn'].map({'Yes': 1, 'No': 0})
             # If conversion isn't straightforward, return error
             return jsonify({"error": "'Churn' column is not numeric (0/1)."}), 400


        # Calculate total revenue hike for churned (1) and retained (0) customers
        # Use fillna(0) in case there are NaN values in the hike column
        revenue_churned = customer_data_raw[customer_data_raw['Churn'] == 1]['OrderAmountHikeFromlastYear'].fillna(0).sum()
        revenue_retained = customer_data_raw[customer_data_raw['Churn'] == 0]['OrderAmountHikeFromlastYear'].fillna(0).sum()

        # Prepare data for frontend (Recharts PieChart usually expects an array of objects)
        # Handle potential negative values if hike can be negative - Pie charts work best with non-negative values.
        # We'll send raw sums, frontend can decide how to display (maybe absolute values or different chart type if sums negative).
        pie_data = [
            {"name": "Revenue Hike Lost (Churned)", "value": revenue_churned},
            {"name": "Revenue Hike Retained", "value": revenue_retained},
        ]
        # Filter out entries where value might be zero or negative if needed for pie chart logic
        # pie_data = [d for d in pie_data if d['value'] > 0]

        return jsonify(pie_data)

    except Exception as e:
        print(f"API Error in /revenue_pie: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/api/graph_data/payment_distribution', methods=['GET'])
def get_payment_distribution_data():
    if not components_loaded or customer_data_raw is None:
        return jsonify({"error": "API components not ready."}), 500
    if 'Churn' not in customer_data_raw.columns or 'PreferredPaymentMode' not in customer_data_raw.columns:
         return jsonify({"error": "Required columns (Churn, PreferredPaymentMode) not found in loaded data."}), 500

    try:
        # Ensure 'Churn' column exists and is numeric (0 or 1)
        if not pd.api.types.is_numeric_dtype(customer_data_raw['Churn']):
             return jsonify({"error": "'Churn' column is not numeric (0/1)."}), 400

        # Calculate counts using crosstab
        # fillna(0) handles cases where a payment mode might only have churned or non-churned users
        counts = pd.crosstab(customer_data_raw['PreferredPaymentMode'], customer_data_raw['Churn']).fillna(0)

        # Rename columns for clarity (assuming 0=No Churn, 1=Churn)
        counts.rename(columns={0: 'noChurn', 1: 'churn'}, inplace=True)

        # Reset index to make 'PreferredPaymentMode' a column
        counts = counts.reset_index()

        # Convert to list of dictionaries suitable for Recharts BarChart
        # Example format: [{'PreferredPaymentMode': 'Card', 'noChurn': 50, 'churn': 10}, ...]
        chart_data = counts.to_dict(orient='records')

        return jsonify(chart_data)

    except Exception as e:
        print(f"API Error in /payment_distribution: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


# --- --- Run the App --- ---
if __name__ == '__main__':
    # Make sure the host is accessible from your frontend (0.0.0.0)
    # Use a port that doesn't conflict (e.g., 5001 if 5000 is common)
>>>>>>> 6beea00e2bacdf6a04b11b2a917af4c0134bb444
    app.run(host='0.0.0.0', port=5001, debug=True) # debug=True is helpful for development