<<<<<<< HEAD
# import pandas as pd
# import numpy as np
# import joblib
# import sklearn # Good practice to import sklearn to ensure custom objects load

# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 1. Configuration ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# # File Paths (Adjust if your files are elsewhere)
# PIPELINE_PATH = 'churn_pipeline.joblib'
# FEATURE_NAMES_PATH = 'processed_feature_names.joblib'
# # This should be the data for customers you want to generate recommendations FOR
# RAW_CUSTOMER_DATA_PATH = 'Cleaned_E_Commerce_Data.csv'

# # Recommendation System Thresholds (ADJUST THESE BASED ON YOUR BUSINESS NEEDS & DATA ANALYSIS)
# HIGH_RISK_THRESHOLD = 0.60  # Probability threshold for churn risk
# INACTIVITY_DAYS_THRESHOLD = 30 # Days since last order considered 'high'
# LOW_SATISFACTION_THRESHOLD = 2  # Satisfaction score considered 'low' (e.g., 1 or 2)
# LOW_TENURE_MONTHS_THRESHOLD = 6 # Tenure in months considered 'low' (new customer)
# NEGATIVE_ORDER_HIKE_THRESHOLD = -5 # Percentage decrease considered significant
# LOW_APP_HOURS_THRESHOLD = 1.5  # Hours spent on app considered 'low'
# HIGH_CHURN_PAYMENT_MODES = ['Debit Card'] # EXAMPLE: List modes identified with high churn in your EDA
# RETENTION_PAYMENT_MODES = ['Cash on Delivery', 'CC'] # EXAMPLE: List modes identified with low churn in your EDA
# TOP_N_FEATURES_TO_CONSIDER = 5 # How many top features importance to check (optional)


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 2. Load Pre-trained Components ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# try:
#     print(f"Loading the full pipeline from: {PIPELINE_PATH}")
#     loaded_pipeline = joblib.load(PIPELINE_PATH)
#     print("Pipeline loaded successfully.")

#     print(f"Loading processed feature names from: {FEATURE_NAMES_PATH}")
#     feature_names_processed = joblib.load(FEATURE_NAMES_PATH)
#     print("Feature names loaded successfully.")

# except FileNotFoundError as e:
#     print(f"Error: File not found. Make sure '{e.filename}' is in the correct directory.")
#     exit()
# except Exception as e:
#     print(f"An error occurred during loading: {e}")
#     exit()

# # --- Extract Preprocessor and Model from Pipeline ---
# try:
#     fitted_preprocessor = loaded_pipeline.named_steps['preprocessor']
#     print("- Fitted preprocessor extracted.")
#     trained_model = loaded_pipeline.named_steps['model']
#     print("- Trained model extracted.")
# except KeyError as e:
#     print(f"\nError: Could not find the step named '{e}' in the loaded pipeline.")
#     print(f"Available steps: {list(loaded_pipeline.named_steps.keys())}")
#     exit()
# except Exception as e:
#     print(f"\nAn unexpected error occurred accessing pipeline steps: {e}")
#     exit()

# # --- Extract Feature Importances (if available) ---
# feature_importances = None
# if hasattr(trained_model, 'feature_importances_'):
#     feature_importances = trained_model.feature_importances_
#     print(f"- Feature importances extracted (Count: {len(feature_importances)}).")
#     # Sanity check length against feature names
#     if len(feature_importances) != len(feature_names_processed):
#         print(f"Warning: Mismatch between feature importance count ({len(feature_importances)}) and processed feature name count ({len(feature_names_processed)}). Importance mapping might be incorrect.")
#         # feature_importances = None # Option: disable importance use
# else:
#     print("- Feature importances not available for this model type.")


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 3. Load Raw Customer Data for Recommendations ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# try:
#     print(f"\nLoading raw customer data from: {RAW_CUSTOMER_DATA_PATH}")
#     customer_data_raw = pd.read_csv(RAW_CUSTOMER_DATA_PATH)
#     print(f"Raw customer data loaded. Shape: {customer_data_raw.shape}")

#     # IMPORTANT CHECK: Ensure CustomerID exists
#     if 'CustomerID' not in customer_data_raw.columns:
#         raise ValueError("The raw customer data CSV must contain a 'CustomerID' column.")

# except FileNotFoundError:
#     print(f"Error: Raw customer data file not found at {RAW_CUSTOMER_DATA_PATH}")
#     exit()
# except ValueError as ve:
#     print(ve)
#     exit()
# except Exception as e:
#     print(f"An error occurred loading raw customer data: {e}")
#     exit()


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 4. Modified Recommendation Generation Function ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# def generate_recommendations_with_cause( # Renamed function for clarity
#     customer_data_raw: pd.DataFrame,
#     churn_model, # The trained model object
#     preprocessor, # The fitted preprocessor object
#     feature_names_processed: list,
#     churn_threshold: float,
#     feature_importances: np.ndarray = None,
#     top_n_features: int = 5
# ):
#     """
#     Generates marketing recommendations for customers predicted to be high churn risk,
#     including the cause (rule triggered).

#     Args:
#         customer_data_raw: DataFrame with raw customer data (including CustomerID).
#         churn_model: Trained classification model with predict_proba method.
#         preprocessor: Fitted preprocessing object/pipeline.
#         feature_names_processed: List of feature names after preprocessing.
#         churn_threshold: Probability above which a customer is considered high-risk.
#         feature_importances: Array of feature importances (optional).
#         top_n_features: Number of top important features to consider (optional).

#     Returns:
#         Dictionary mapping CustomerID to a list of dictionaries,
#         where each dict has {'Cause': str, 'Recommendation': str}.
#     """

#     recommendations = {}
#     if 'CustomerID' not in customer_data_raw.columns:
#         raise ValueError("customer_data_raw must include a 'CustomerID' column.")

#     # Ensure CustomerID is kept for mapping results
#     customer_ids = customer_data_raw['CustomerID']
#     # Prepare data for prediction (excluding CustomerID if present)
#     X_raw = customer_data_raw.drop('CustomerID', axis=1, errors='ignore')

#     print("Preprocessing customer data...")
#     try:
#         X_processed = preprocessor.transform(X_raw)
#     except Exception as e:
#         print(f"Error during preprocessing: {e}")
#         return {} # Return empty if preprocessing fails

#     print("Predicting churn probabilities...")
#     churn_probabilities = churn_model.predict_proba(X_processed)[:, 1] # Probability of class 1 (Churn)

#     # Store probabilities alongside IDs for potential sorting later if needed
#     churn_predictions_df = pd.DataFrame({
#         'CustomerID': customer_ids,
#         'ChurnProbability': churn_probabilities,
#         'OriginalIndex': customer_ids.index # Keep track of original index
#     })

#     # Identify high-risk customers
#     high_risk_df = churn_predictions_df[churn_predictions_df['ChurnProbability'] >= churn_threshold].copy()
#     high_risk_df.sort_values(by='ChurnProbability', ascending=False, inplace=True) # Sort by probability
#     print(f"Found {len(high_risk_df)} high-risk customers (Prob >= {churn_threshold}).")

#     # Optional: Identify top N important features
#     important_feature_names = []
#     if feature_importances is not None and len(feature_importances) == len(feature_names_processed):
#         indices = np.argsort(feature_importances)[::-1]
#         important_feature_names = [feature_names_processed[i] for i in indices[:top_n_features]]
#         print(f"\nTop {top_n_features} features driving predictions (based on model):")
#         for i in indices[:top_n_features]:
#             print(f"- {feature_names_processed[i]}: {feature_importances[i]:.4f}")
#     else:
#         print("\nFeature importances not available or mismatched. Skipping importance-based insights.")


#     print("\nGenerating recommendations for high-risk customers...")
#     # Iterate through high-risk customers based on their original index
#     for _, high_risk_row in high_risk_df.iterrows():
#         customer_id = high_risk_row['CustomerID']
#         original_idx = high_risk_row['OriginalIndex']
#         customer_record = customer_data_raw.loc[original_idx] # Get the raw data row
#         customer_recs_list = [] # List to store {'Cause': ..., 'Recommendation': ...} dicts

#         # --- Apply Rule-Based Logic ---

#         # 1. Based on DaySinceLastOrder
#         if 'DaySinceLastOrder' in customer_record and pd.notna(customer_record['DaySinceLastOrder']) and customer_record['DaySinceLastOrder'] > INACTIVITY_DAYS_THRESHOLD:
#             cause = f"High Inactivity ({int(customer_record['DaySinceLastOrder'])} days)"
#             rec = f"Re-engagement Campaign: Offer special discount/reminder."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 2. Based on SatisfactionScore
#         if 'SatisfactionScore' in customer_record and pd.notna(customer_record['SatisfactionScore']) and customer_record['SatisfactionScore'] <= LOW_SATISFACTION_THRESHOLD:
#             cause = f"Low Satisfaction Score ({int(customer_record['SatisfactionScore'])})"
#             rec = "Proactive support outreach, feedback survey, or goodwill gesture needed."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 3. Based on Tenure
#         if 'Tenure' in customer_record and pd.notna(customer_record['Tenure']) and customer_record['Tenure'] <= LOW_TENURE_MONTHS_THRESHOLD:
#             cause = f"Low Tenure ({int(customer_record['Tenure'])} months)"
#             rec = "Focus on onboarding, tutorials, first-purchase offers, or welcome check-in."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 4. Based on CashbackAmount / CouponUsed Importance
#         if 'CashbackAmount' in important_feature_names or 'CouponUsed' in important_feature_names:
#             cause = "High Sensitivity to Incentives (Cashback/Coupon)"
#             rec = "Check usage of cashback/coupons. Consider targeted loyalty offer."
#             # Avoid adding duplicate if specific cashback rule also triggers
#             if not any(d['Cause'].startswith('Review Incentives') for d in customer_recs_list):
#                  customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
#         elif 'CashbackAmount' in customer_record and pd.notna(customer_record['CashbackAmount']): # Basic check even if not top feature
#              # Add a more specific rule perhaps? E.g., if cashback is below average?
#              # For now, just a generic check if present
#              cause = f"Review Incentives (Cashback: {customer_record['CashbackAmount']:.2f})"
#              rec = "Consider personalized coupon/cashback if activity is low or decreasing."
#              # Avoid adding duplicates if the importance rule triggered
#              if not any(d['Cause'].startswith('High Sensitivity to Incentives') for d in customer_recs_list):
#                  customer_recs_list.append({'Cause': cause, 'Recommendation': rec})


#         # 5. Based on Payment Mode
#         if 'PreferredPaymentMode' in customer_record and pd.notna(customer_record['PreferredPaymentMode']):
#             pm = customer_record['PreferredPaymentMode']
#             if pm in HIGH_CHURN_PAYMENT_MODES:
#                 cause = f"Using High-Risk Payment Mode ({pm})"
#                 rec = f"Investigate friction. Consider incentive to switch to modes like {RETENTION_PAYMENT_MODES}."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
#             elif pm not in RETENTION_PAYMENT_MODES:
#                 cause = f"Not Using Preferred Payment Mode ({pm})"
#                 rec = f"Encourage trial of preferred modes ({RETENTION_PAYMENT_MODES}) for potentially better retention."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 6. Based on Personalized Outreach (Order Category)
#         if 'PreferedOrderCat' in customer_record and pd.notna(customer_record['PreferedOrderCat']):
#             fav_cat = customer_record['PreferedOrderCat']
#             cause = f"Opportunity: Preferred Category ('{fav_cat}')"
#             rec = f"Target with promotion related to '{fav_cat}'. E.g., '10% off your next {fav_cat} order!'"
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec}) # This is more an opportunity

#         # 7. Based on OrderAmountHikeFromlastYear
#         if 'OrderAmountHikeFromlastYear' in customer_record and pd.notna(customer_record['OrderAmountHikeFromlastYear']):
#             hike = customer_record['OrderAmountHikeFromlastYear']
#             if hike <= NEGATIVE_ORDER_HIKE_THRESHOLD:
#                 cause = f"Decreased Spending ({hike:.1f}%)"
#                 rec = "Target with value bundles or special promotions."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
#             elif -5 < hike < 5: # Example threshold for stagnant
#                 cause = "Stagnant Spending"
#                 rec = "Introduce relevant new products or categories based on purchase history."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 8. Based on App/Website Engagement
#         if 'HourSpendOnApp' in customer_record and pd.notna(customer_record['HourSpendOnApp']) and customer_record['HourSpendOnApp'] < LOW_APP_HOURS_THRESHOLD:
#             cause = f"Low App Engagement ({customer_record['HourSpendOnApp']:.1f} hrs)"
#             rec = "Investigate app usability/content. Use targeted in-app messages for features/offers."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # Add more rules as needed...

#         # Assign collected recommendations to the customer ID
#         if customer_recs_list: # Only add if there are recommendations
#             recommendations[customer_id] = customer_recs_list
#         else:
#             # Store the generic message in the same format
#             recommendations[customer_id] = [{'Cause': 'High Churn Risk (Generic)', 'Recommendation': "No specific rule triggered. Review manually or broaden campaign."}]


#     print("\nRecommendation generation complete.")
#     return recommendations

# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 5. Generate and Format Output ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# recommendations_output_dict = {}
# if customer_data_raw is not None:
#     recommendations_output_dict = generate_recommendations_with_cause(
#         customer_data_raw=customer_data_raw,
#         churn_model=trained_model,
#         preprocessor=fitted_preprocessor,
#         feature_names_processed=feature_names_processed,
#         churn_threshold=HIGH_RISK_THRESHOLD,
#         feature_importances=feature_importances,
#         top_n_features=TOP_N_FEATURES_TO_CONSIDER
#     )

# # --- Create DataFrame for Output ---
# output_rows = []
# if recommendations_output_dict:
#     print(f"\nFormatting recommendations for {len(recommendations_output_dict)} high-risk customers...")
#     for customer_id, rec_list in recommendations_output_dict.items():
#         # Since one customer can have multiple causes/recommendations, create a row for each
#         for rec_dict in rec_list:
#             output_rows.append({
#                 'CustomerID': customer_id,
#                 'Cause of High Churn Risk': rec_dict['Cause'],
#                 'Recommendation': rec_dict['Recommendation']
#             })

#     output_df = pd.DataFrame(output_rows)

#     # --- Display Top 50 Rows ---
#     print("\n--- --- --- Top 50 Marketing Recommendations --- --- ---")
#     # Note: The customers are already sorted by churn probability descending due to changes above.
#     # We take the top 100 *rows* which might include multiple entries for the highest-risk customers.
#     top_50_df = output_df.head(50)

#     if not top_50_df.empty:
#          # Optional: Display the DataFrame directly
#         print(top_50_df.to_string()) # Use to_string to print full DF without truncation

#         # Optional: Save to CSV
#         # top_100_df.to_csv("top_100_churn_recommendations.csv", index=False)
#         # print("\nTop 100 recommendations saved to 'top_100_churn_recommendations.csv'")
#     else:
#         print("\nGenerated DataFrame is empty (perhaps no high-risk customers found).")

# else:
#     # Provide more context if no recommendations were generated
#     print("\nNo recommendations generated.")
#     print(f"Possible reasons:")
#     print(f"  - No customers found with churn probability >= {HIGH_RISK_THRESHOLD}.")
#     print(f"  - Check the 'Predicting churn probabilities...' step output. If max probability is low, consider lowering the threshold or retraining the model.")
#     print(f"  - Ensure the input data file ('{RAW_CUSTOMER_DATA_PATH}') contains relevant customer data.")


=======
# import pandas as pd
# import numpy as np
# import joblib
# import sklearn # Good practice to import sklearn to ensure custom objects load

# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 1. Configuration ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# # File Paths (Adjust if your files are elsewhere)
# PIPELINE_PATH = 'churn_pipeline.joblib'
# FEATURE_NAMES_PATH = 'processed_feature_names.joblib'
# # This should be the data for customers you want to generate recommendations FOR
# RAW_CUSTOMER_DATA_PATH = 'Cleaned_E_Commerce_Data.csv'

# # Recommendation System Thresholds (ADJUST THESE BASED ON YOUR BUSINESS NEEDS & DATA ANALYSIS)
# HIGH_RISK_THRESHOLD = 0.60  # Probability threshold for churn risk
# INACTIVITY_DAYS_THRESHOLD = 30 # Days since last order considered 'high'
# LOW_SATISFACTION_THRESHOLD = 2  # Satisfaction score considered 'low' (e.g., 1 or 2)
# LOW_TENURE_MONTHS_THRESHOLD = 6 # Tenure in months considered 'low' (new customer)
# NEGATIVE_ORDER_HIKE_THRESHOLD = -5 # Percentage decrease considered significant
# LOW_APP_HOURS_THRESHOLD = 1.5  # Hours spent on app considered 'low'
# HIGH_CHURN_PAYMENT_MODES = ['Debit Card'] # EXAMPLE: List modes identified with high churn in your EDA
# RETENTION_PAYMENT_MODES = ['Cash on Delivery', 'CC'] # EXAMPLE: List modes identified with low churn in your EDA
# TOP_N_FEATURES_TO_CONSIDER = 5 # How many top features importance to check (optional)


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 2. Load Pre-trained Components ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# try:
#     print(f"Loading the full pipeline from: {PIPELINE_PATH}")
#     loaded_pipeline = joblib.load(PIPELINE_PATH)
#     print("Pipeline loaded successfully.")

#     print(f"Loading processed feature names from: {FEATURE_NAMES_PATH}")
#     feature_names_processed = joblib.load(FEATURE_NAMES_PATH)
#     print("Feature names loaded successfully.")

# except FileNotFoundError as e:
#     print(f"Error: File not found. Make sure '{e.filename}' is in the correct directory.")
#     exit()
# except Exception as e:
#     print(f"An error occurred during loading: {e}")
#     exit()

# # --- Extract Preprocessor and Model from Pipeline ---
# try:
#     fitted_preprocessor = loaded_pipeline.named_steps['preprocessor']
#     print("- Fitted preprocessor extracted.")
#     trained_model = loaded_pipeline.named_steps['model']
#     print("- Trained model extracted.")
# except KeyError as e:
#     print(f"\nError: Could not find the step named '{e}' in the loaded pipeline.")
#     print(f"Available steps: {list(loaded_pipeline.named_steps.keys())}")
#     exit()
# except Exception as e:
#     print(f"\nAn unexpected error occurred accessing pipeline steps: {e}")
#     exit()

# # --- Extract Feature Importances (if available) ---
# feature_importances = None
# if hasattr(trained_model, 'feature_importances_'):
#     feature_importances = trained_model.feature_importances_
#     print(f"- Feature importances extracted (Count: {len(feature_importances)}).")
#     # Sanity check length against feature names
#     if len(feature_importances) != len(feature_names_processed):
#         print(f"Warning: Mismatch between feature importance count ({len(feature_importances)}) and processed feature name count ({len(feature_names_processed)}). Importance mapping might be incorrect.")
#         # feature_importances = None # Option: disable importance use
# else:
#     print("- Feature importances not available for this model type.")


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 3. Load Raw Customer Data for Recommendations ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# try:
#     print(f"\nLoading raw customer data from: {RAW_CUSTOMER_DATA_PATH}")
#     customer_data_raw = pd.read_csv(RAW_CUSTOMER_DATA_PATH)
#     print(f"Raw customer data loaded. Shape: {customer_data_raw.shape}")

#     # IMPORTANT CHECK: Ensure CustomerID exists
#     if 'CustomerID' not in customer_data_raw.columns:
#         raise ValueError("The raw customer data CSV must contain a 'CustomerID' column.")

# except FileNotFoundError:
#     print(f"Error: Raw customer data file not found at {RAW_CUSTOMER_DATA_PATH}")
#     exit()
# except ValueError as ve:
#     print(ve)
#     exit()
# except Exception as e:
#     print(f"An error occurred loading raw customer data: {e}")
#     exit()


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 4. Modified Recommendation Generation Function ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# def generate_recommendations_with_cause( # Renamed function for clarity
#     customer_data_raw: pd.DataFrame,
#     churn_model, # The trained model object
#     preprocessor, # The fitted preprocessor object
#     feature_names_processed: list,
#     churn_threshold: float,
#     feature_importances: np.ndarray = None,
#     top_n_features: int = 5
# ):
#     """
#     Generates marketing recommendations for customers predicted to be high churn risk,
#     including the cause (rule triggered).

#     Args:
#         customer_data_raw: DataFrame with raw customer data (including CustomerID).
#         churn_model: Trained classification model with predict_proba method.
#         preprocessor: Fitted preprocessing object/pipeline.
#         feature_names_processed: List of feature names after preprocessing.
#         churn_threshold: Probability above which a customer is considered high-risk.
#         feature_importances: Array of feature importances (optional).
#         top_n_features: Number of top important features to consider (optional).

#     Returns:
#         Dictionary mapping CustomerID to a list of dictionaries,
#         where each dict has {'Cause': str, 'Recommendation': str}.
#     """

#     recommendations = {}
#     if 'CustomerID' not in customer_data_raw.columns:
#         raise ValueError("customer_data_raw must include a 'CustomerID' column.")

#     # Ensure CustomerID is kept for mapping results
#     customer_ids = customer_data_raw['CustomerID']
#     # Prepare data for prediction (excluding CustomerID if present)
#     X_raw = customer_data_raw.drop('CustomerID', axis=1, errors='ignore')

#     print("Preprocessing customer data...")
#     try:
#         X_processed = preprocessor.transform(X_raw)
#     except Exception as e:
#         print(f"Error during preprocessing: {e}")
#         return {} # Return empty if preprocessing fails

#     print("Predicting churn probabilities...")
#     churn_probabilities = churn_model.predict_proba(X_processed)[:, 1] # Probability of class 1 (Churn)

#     # Store probabilities alongside IDs for potential sorting later if needed
#     churn_predictions_df = pd.DataFrame({
#         'CustomerID': customer_ids,
#         'ChurnProbability': churn_probabilities,
#         'OriginalIndex': customer_ids.index # Keep track of original index
#     })

#     # Identify high-risk customers
#     high_risk_df = churn_predictions_df[churn_predictions_df['ChurnProbability'] >= churn_threshold].copy()
#     high_risk_df.sort_values(by='ChurnProbability', ascending=False, inplace=True) # Sort by probability
#     print(f"Found {len(high_risk_df)} high-risk customers (Prob >= {churn_threshold}).")

#     # Optional: Identify top N important features
#     important_feature_names = []
#     if feature_importances is not None and len(feature_importances) == len(feature_names_processed):
#         indices = np.argsort(feature_importances)[::-1]
#         important_feature_names = [feature_names_processed[i] for i in indices[:top_n_features]]
#         print(f"\nTop {top_n_features} features driving predictions (based on model):")
#         for i in indices[:top_n_features]:
#             print(f"- {feature_names_processed[i]}: {feature_importances[i]:.4f}")
#     else:
#         print("\nFeature importances not available or mismatched. Skipping importance-based insights.")


#     print("\nGenerating recommendations for high-risk customers...")
#     # Iterate through high-risk customers based on their original index
#     for _, high_risk_row in high_risk_df.iterrows():
#         customer_id = high_risk_row['CustomerID']
#         original_idx = high_risk_row['OriginalIndex']
#         customer_record = customer_data_raw.loc[original_idx] # Get the raw data row
#         customer_recs_list = [] # List to store {'Cause': ..., 'Recommendation': ...} dicts

#         # --- Apply Rule-Based Logic ---

#         # 1. Based on DaySinceLastOrder
#         if 'DaySinceLastOrder' in customer_record and pd.notna(customer_record['DaySinceLastOrder']) and customer_record['DaySinceLastOrder'] > INACTIVITY_DAYS_THRESHOLD:
#             cause = f"High Inactivity ({int(customer_record['DaySinceLastOrder'])} days)"
#             rec = f"Re-engagement Campaign: Offer special discount/reminder."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 2. Based on SatisfactionScore
#         if 'SatisfactionScore' in customer_record and pd.notna(customer_record['SatisfactionScore']) and customer_record['SatisfactionScore'] <= LOW_SATISFACTION_THRESHOLD:
#             cause = f"Low Satisfaction Score ({int(customer_record['SatisfactionScore'])})"
#             rec = "Proactive support outreach, feedback survey, or goodwill gesture needed."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 3. Based on Tenure
#         if 'Tenure' in customer_record and pd.notna(customer_record['Tenure']) and customer_record['Tenure'] <= LOW_TENURE_MONTHS_THRESHOLD:
#             cause = f"Low Tenure ({int(customer_record['Tenure'])} months)"
#             rec = "Focus on onboarding, tutorials, first-purchase offers, or welcome check-in."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 4. Based on CashbackAmount / CouponUsed Importance
#         if 'CashbackAmount' in important_feature_names or 'CouponUsed' in important_feature_names:
#             cause = "High Sensitivity to Incentives (Cashback/Coupon)"
#             rec = "Check usage of cashback/coupons. Consider targeted loyalty offer."
#             # Avoid adding duplicate if specific cashback rule also triggers
#             if not any(d['Cause'].startswith('Review Incentives') for d in customer_recs_list):
#                  customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
#         elif 'CashbackAmount' in customer_record and pd.notna(customer_record['CashbackAmount']): # Basic check even if not top feature
#              # Add a more specific rule perhaps? E.g., if cashback is below average?
#              # For now, just a generic check if present
#              cause = f"Review Incentives (Cashback: {customer_record['CashbackAmount']:.2f})"
#              rec = "Consider personalized coupon/cashback if activity is low or decreasing."
#              # Avoid adding duplicates if the importance rule triggered
#              if not any(d['Cause'].startswith('High Sensitivity to Incentives') for d in customer_recs_list):
#                  customer_recs_list.append({'Cause': cause, 'Recommendation': rec})


#         # 5. Based on Payment Mode
#         if 'PreferredPaymentMode' in customer_record and pd.notna(customer_record['PreferredPaymentMode']):
#             pm = customer_record['PreferredPaymentMode']
#             if pm in HIGH_CHURN_PAYMENT_MODES:
#                 cause = f"Using High-Risk Payment Mode ({pm})"
#                 rec = f"Investigate friction. Consider incentive to switch to modes like {RETENTION_PAYMENT_MODES}."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
#             elif pm not in RETENTION_PAYMENT_MODES:
#                 cause = f"Not Using Preferred Payment Mode ({pm})"
#                 rec = f"Encourage trial of preferred modes ({RETENTION_PAYMENT_MODES}) for potentially better retention."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 6. Based on Personalized Outreach (Order Category)
#         if 'PreferedOrderCat' in customer_record and pd.notna(customer_record['PreferedOrderCat']):
#             fav_cat = customer_record['PreferedOrderCat']
#             cause = f"Opportunity: Preferred Category ('{fav_cat}')"
#             rec = f"Target with promotion related to '{fav_cat}'. E.g., '10% off your next {fav_cat} order!'"
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec}) # This is more an opportunity

#         # 7. Based on OrderAmountHikeFromlastYear
#         if 'OrderAmountHikeFromlastYear' in customer_record and pd.notna(customer_record['OrderAmountHikeFromlastYear']):
#             hike = customer_record['OrderAmountHikeFromlastYear']
#             if hike <= NEGATIVE_ORDER_HIKE_THRESHOLD:
#                 cause = f"Decreased Spending ({hike:.1f}%)"
#                 rec = "Target with value bundles or special promotions."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})
#             elif -5 < hike < 5: # Example threshold for stagnant
#                 cause = "Stagnant Spending"
#                 rec = "Introduce relevant new products or categories based on purchase history."
#                 customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # 8. Based on App/Website Engagement
#         if 'HourSpendOnApp' in customer_record and pd.notna(customer_record['HourSpendOnApp']) and customer_record['HourSpendOnApp'] < LOW_APP_HOURS_THRESHOLD:
#             cause = f"Low App Engagement ({customer_record['HourSpendOnApp']:.1f} hrs)"
#             rec = "Investigate app usability/content. Use targeted in-app messages for features/offers."
#             customer_recs_list.append({'Cause': cause, 'Recommendation': rec})

#         # Add more rules as needed...

#         # Assign collected recommendations to the customer ID
#         if customer_recs_list: # Only add if there are recommendations
#             recommendations[customer_id] = customer_recs_list
#         else:
#             # Store the generic message in the same format
#             recommendations[customer_id] = [{'Cause': 'High Churn Risk (Generic)', 'Recommendation': "No specific rule triggered. Review manually or broaden campaign."}]


#     print("\nRecommendation generation complete.")
#     return recommendations

# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # --- 5. Generate and Format Output ---
# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# recommendations_output_dict = {}
# if customer_data_raw is not None:
#     recommendations_output_dict = generate_recommendations_with_cause(
#         customer_data_raw=customer_data_raw,
#         churn_model=trained_model,
#         preprocessor=fitted_preprocessor,
#         feature_names_processed=feature_names_processed,
#         churn_threshold=HIGH_RISK_THRESHOLD,
#         feature_importances=feature_importances,
#         top_n_features=TOP_N_FEATURES_TO_CONSIDER
#     )

# # --- Create DataFrame for Output ---
# output_rows = []
# if recommendations_output_dict:
#     print(f"\nFormatting recommendations for {len(recommendations_output_dict)} high-risk customers...")
#     for customer_id, rec_list in recommendations_output_dict.items():
#         # Since one customer can have multiple causes/recommendations, create a row for each
#         for rec_dict in rec_list:
#             output_rows.append({
#                 'CustomerID': customer_id,
#                 'Cause of High Churn Risk': rec_dict['Cause'],
#                 'Recommendation': rec_dict['Recommendation']
#             })

#     output_df = pd.DataFrame(output_rows)

#     # --- Display Top 50 Rows ---
#     print("\n--- --- --- Top 50 Marketing Recommendations --- --- ---")
#     # Note: The customers are already sorted by churn probability descending due to changes above.
#     # We take the top 100 *rows* which might include multiple entries for the highest-risk customers.
#     top_50_df = output_df.head(50)

#     if not top_50_df.empty:
#          # Optional: Display the DataFrame directly
#         print(top_50_df.to_string()) # Use to_string to print full DF without truncation

#         # Optional: Save to CSV
#         # top_100_df.to_csv("top_100_churn_recommendations.csv", index=False)
#         # print("\nTop 100 recommendations saved to 'top_100_churn_recommendations.csv'")
#     else:
#         print("\nGenerated DataFrame is empty (perhaps no high-risk customers found).")

# else:
#     # Provide more context if no recommendations were generated
#     print("\nNo recommendations generated.")
#     print(f"Possible reasons:")
#     print(f"  - No customers found with churn probability >= {HIGH_RISK_THRESHOLD}.")
#     print(f"  - Check the 'Predicting churn probabilities...' step output. If max probability is low, consider lowering the threshold or retraining the model.")
#     print(f"  - Ensure the input data file ('{RAW_CUSTOMER_DATA_PATH}') contains relevant customer data.")


>>>>>>> 6beea00e2bacdf6a04b11b2a917af4c0134bb444
# print("\n--- --- --- End of Script --- --- ---")