import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import optuna
import pickle

# Load the dataset
df = pd.read_csv('last.csv')
df = df.drop(["Unnamed: 0", "timestamp", "username", "category_name", "count_caption", "estimate_likes", "estimate_likes_media_type"], axis=1)
# df['estimate_likes_media_type'] = np.maximum(df['estimate_likes_media_type'], df['min_likes_media_type'])
print("Data shape:", df.shape)


# --- Drop median features ---
df = df.drop(['median_likes', 'median_comments', 'median_likes_media_type', 'median_comments_media_type'], axis=1)

# --- Binary Categorical Encoding ---
binary_features = ['is_verified', 'is_business_account', 'is_holiday', 'is_professional_account', 'is_lottery', 'media_type_VIDEO']
encoder = LabelEncoder()
for feature in binary_features:
    df[feature] = encoder.fit_transform(df[feature])

df['mean_max_likes_ratio'] = df['mean_likes'] / (df['max_likes'] + 1e-5)
df['comments_per_follower'] = df['comments_count'] / (df['follower_count'] + 1e-5)
df['t_1_to_mean_likes'] = df['t_1_likes']/(df['mean_likes']+ 1e-5)
df['t_1_to_follower'] = df['t_1_likes']/(df['follower_count']+ 1e-5)
df['comment_to_post'] = df['comments_count']/(df['post_count'] + 1e-5)

# More Interactions:
df['t_1_t_2'] = df['t_1_likes'] * df['t_2_likes']

# Function to impute outliers using IQR boundaries
def impute_outliers(df, features, iqr_multiplier=1.5):
    df_imputed = df.copy()  # to not make changes to df outside the function
    for feature in features:
        Q1 = df[feature].quantile(0.01)
        Q3 = df[feature].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        df_imputed[feature] = np.where(df_imputed[feature] < lower_bound, lower_bound, df_imputed[feature]) #lower bound
        df_imputed[feature] = np.where(df_imputed[feature] > upper_bound, upper_bound, df_imputed[feature]) #upper bound
    return df_imputed


# --- Define numeric features ---
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('is_verified') #remove is_verified from numerical_features list
numerical_features.remove('like_count') # remove target from outlier detection

# --- Impute outliers in training data using IQR boundaries ---
impute_outliers_flag = True #set to false if you dont want outlier imputations
if impute_outliers_flag:
  df_imputed = impute_outliers(df, numerical_features)
else:
    df_imputed = df.copy()


# --- Split into verified and non-verified data ---
df_verified = df_imputed[df_imputed['is_verified'] == 1]
df_non_verified = df_imputed[df_imputed['is_verified'] == 0]


# Function to train model and make predictions
def train_and_predict(df, is_verified_label):
    X = df.drop(columns='like_count')
    y = df['like_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=15)

    # Optuna Hyperparameter Tuning
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'objective': 'reg:gamma',
            'seed': 8,
            'eval_metric': 'mae',
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"Best Parameters for is_verified={is_verified_label}:", best_params)


    # Train the final model
    best_model = XGBRegressor(**best_params, objective='reg:gamma', seed=8, eval_metric='mae')
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    # Predict
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Ensure non-negative
    y_train_pred = np.maximum(y_train_pred, 0)
    y_test_pred = np.maximum(y_test_pred, 0)

    # Calculate MAE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Train MAE for is_verified={is_verified_label}: {train_mae:.4f}")
    print(f"Test MAE for is_verified={is_verified_label}: {test_mae:.4f}")

     # Save Predictions
    df_test = df.loc[X_test.index].copy()  #to avoid setting with copy
    df_test['predicted_like_count'] = y_test_pred
    df_test.to_csv(f'final_predictions_is_verified_{is_verified_label}.csv', index=False)
    print(f"Final predictions saved to 'final_predictions_is_verified_{is_verified_label}.csv'.")

    # Save Model
    with open(f'model_is_verified_{is_verified_label}.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print(f"Model saved as 'model_is_verified_{is_verified_label}.pkl'")

    # Print feature importance
    print(f"Feature importance for is_verified={is_verified_label}:")
    feature_importances = sorted(best_model.feature_importances_)
    feature_names = X.columns
    for name, importance in zip(feature_names, feature_importances):
      print(f"{name}: {importance:.4f}")

    return df_test


# Train and Predict for verified users
df_test_verified = train_and_predict(df_verified, 1)

# # Train and Predict for non-verified users
df_test_non_verified = train_and_predict(df_non_verified, 0)