# ============================================================================
# IMDB SCORE PREDICTION - OPTIMIZED WITH BEST 12 FEATURES (NO NULLS)
# Copy and paste this entire cell into Colab and run it!
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: UPLOAD YOUR FILE
# ============================================================================
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Load data
df = pd.read_csv(filename)
print(f"✅ Loaded {df.shape[0]} movies with {df.shape[1]} features\n")

# ============================================================================
# STEP 2: CLEAN TARGET VARIABLE
# ============================================================================
# Keep only rows with valid IMDB scores
df = df.dropna(subset=['imdb_score'])
df = df[(df['imdb_score'] >= 0) & (df['imdb_score'] <= 10)]
print(f"✅ After cleaning target: {df.shape[0]} movies remaining\n")

# ============================================================================
# STEP 3: COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================
print("🔧 Creating features...\n")

# Fill missing values for feature engineering
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'imdb_score':
        df[col] = df[col].fillna(df[col].median())

# Categorical features
for col in ['content_rating', 'country', 'language', 'color', 'genres']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

# Movie age
if 'title_year' in df.columns:
    df['title_year'] = df['title_year'].fillna(df['title_year'].median())
    df['movie_age'] = 2026 - df['title_year']

# Genre count and binary features
if 'genres' in df.columns:
    df['num_genres'] = df['genres'].apply(lambda x: len(str(x).split('|')))
    df['is_Action'] = df['genres'].str.contains('Action', na=False).astype(int)
    df['is_Comedy'] = df['genres'].str.contains('Comedy', na=False).astype(int)
    df['is_Drama'] = df['genres'].str.contains('Drama', na=False).astype(int)
    df['is_Thriller'] = df['genres'].str.contains('Thriller', na=False).astype(int)
    df['is_Romance'] = df['genres'].str.contains('Romance', na=False).astype(int)
    df['is_Horror'] = df['genres'].str.contains('Horror', na=False).astype(int)

# Log transforms for skewed distributions
for col in ['gross', 'budget', 'movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes']:
    if col in df.columns:
        df[f'{col}_log'] = np.log1p(df[col])

# Ratio features (often highly predictive)
if 'budget' in df.columns and 'gross' in df.columns:
    df['profit_ratio'] = np.where(df['budget'] > 0, df['gross'] / (df['budget'] + 1), 0)
    df['roi'] = np.where(df['budget'] > 0, (df['gross'] - df['budget']) / (df['budget'] + 1), 0)

if 'num_critic_for_reviews' in df.columns and 'num_user_for_reviews' in df.columns:
    df['critic_user_ratio'] = np.where(
        df['num_user_for_reviews'] > 0,
        df['num_critic_for_reviews'] / (df['num_user_for_reviews'] + 1),
        0
    )

if 'movie_facebook_likes' in df.columns and 'cast_total_facebook_likes' in df.columns:
    df['movie_cast_likes_ratio'] = np.where(
        df['cast_total_facebook_likes'] > 0,
        df['movie_facebook_likes'] / (df['cast_total_facebook_likes'] + 1),
        0
    )

# Actor likes sum
if all(col in df.columns for col in ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes']):
    df['total_actor_likes'] = (
        df['actor_1_facebook_likes'] + 
        df['actor_2_facebook_likes'] + 
        df['actor_3_facebook_likes']
    )
    df['total_actor_likes_log'] = np.log1p(df['total_actor_likes'])

# Encode categorical
from sklearn.preprocessing import LabelEncoder
for col in ['content_rating', 'color', 'language', 'country']:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

print(f"✅ Features created: {df.shape[1]} total features\n")

# ============================================================================
# STEP 4: SELECT BEST 12 FEATURES (WITH NO NULLS)
# ============================================================================
print("🎯 Selecting best 12 features with zero nulls...\n")

# All potential features
candidate_features = [
    # Original numeric features
    'num_critic_for_reviews', 'duration', 'director_facebook_likes',
    'actor_3_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes',
    'gross', 'num_user_for_reviews', 'budget', 'aspect_ratio',
    'movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes',
    
    # Engineered features
    'movie_age', 'num_genres', 'total_actor_likes',
    
    # Log transforms
    'gross_log', 'budget_log', 'movie_facebook_likes_log', 
    'num_voted_users_log', 'cast_total_facebook_likes_log', 'total_actor_likes_log',
    
    # Ratios
    'profit_ratio', 'roi', 'critic_user_ratio', 'movie_cast_likes_ratio',
    
    # Genre binary
    'is_Action', 'is_Comedy', 'is_Drama', 'is_Thriller', 'is_Romance', 'is_Horror',
    
    # Encoded categorical
    'content_rating_encoded', 'color_encoded', 'language_encoded', 'country_encoded'
]

# Filter: only features that exist and have no nulls
available_features = []
for feature in candidate_features:
    if feature in df.columns:
        null_count = df[feature].isnull().sum()
        if null_count == 0:
            available_features.append(feature)

print(f"✅ Found {len(available_features)} features with zero nulls\n")

# Calculate feature importance using mutual information
if len(available_features) > 12:
    print("🔍 Calculating feature importance to select top 12...\n")
    
    X_temp = df[available_features]
    y_temp = df['imdb_score']
    
    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X_temp, y_temp, random_state=42)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    # Select top 12 features
    best_12_features = feature_importance.head(12)['feature'].tolist()
    
    print("="*70)
    print("TOP 12 SELECTED FEATURES (RANKED BY IMPORTANCE)")
    print("="*70)
    for i, (idx, row) in enumerate(feature_importance.head(12).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:35s} Score: {row['importance']:.4f}")
    print("="*70 + "\n")
    
else:
    best_12_features = available_features[:12]
    print(f"⚠️  Only {len(available_features)} features available. Using all of them.\n")

# ============================================================================
# STEP 5: VERIFY NO NULLS IN SELECTED FEATURES
# ============================================================================
print("✅ Verifying selected features have zero nulls...\n")

for feature in best_12_features:
    null_count = df[feature].isnull().sum()
    print(f"   {feature:35s} → {null_count} nulls")

total_nulls = df[best_12_features].isnull().sum().sum()
print(f"\n✅ Total nulls in selected features: {total_nulls} (should be 0)\n")

if total_nulls > 0:
    print("❌ ERROR: Found nulls in selected features. This should not happen!")
    raise ValueError("Null values detected in selected features")

# ============================================================================
# STEP 6: PREPARE DATA FOR MODELING
# ============================================================================

X = df[best_12_features]
y = df['imdb_score']

# Verify shapes
print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Test samples: {len(X_test)}\n")

# ============================================================================
# STEP 7: TRAIN ENSEMBLE MODEL (OPTIMIZED FOR 80%+ ACCURACY)
# ============================================================================
print("🚀 Training optimized ensemble model...\n")

# Model 1: Random Forest (optimized hyperparameters)
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Model 2: Gradient Boosting (optimized hyperparameters)
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

# Train models
print("   Training Random Forest...")
rf.fit(X_train_scaled, y_train)
print("   ✅ Random Forest trained")

print("   Training Gradient Boosting...")
gb.fit(X_train_scaled, y_train)
print("   ✅ Gradient Boosting trained\n")

# Make predictions (ensemble average)
y_pred_train_rf = rf.predict(X_train_scaled)
y_pred_train_gb = gb.predict(X_train_scaled)
y_pred_train = (y_pred_train_rf + y_pred_train_gb) / 2

y_pred_test_rf = rf.predict(X_test_scaled)
y_pred_test_gb = gb.predict(X_test_scaled)
y_pred_test = (y_pred_test_rf + y_pred_test_gb) / 2

# Clip predictions to valid range [0, 10]
y_pred_train = np.clip(y_pred_train, 0, 10)
y_pred_test = np.clip(y_pred_test, 0, 10)

# ============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("="*70)
print("MODEL PERFORMANCE REPORT")
print("="*70)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Display results
print(f"\n📈 TRAINING SET METRICS:")
print(f"   R² Score (Accuracy): {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"   MAE: {train_mae:.4f} IMDB points")
print(f"   RMSE: {train_rmse:.4f} IMDB points")

print(f"\n📈 TEST SET METRICS:")
print(f"   R² Score (Accuracy): {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"   MAE: {test_mae:.4f} IMDB points")
print(f"   RMSE: {test_rmse:.4f} IMDB points")

# Check if target achieved
print("\n" + "="*70)
if test_r2 >= 0.80:
    print(f"✅ SUCCESS! Achieved {test_r2*100:.2f}% accuracy (Target: 80%)")
    print(f"✅ Your model predicts IMDB scores within ±{test_mae:.3f} points on average")
else:
    print(f"📊 Current accuracy: {test_r2*100:.2f}% (Target: 80%)")
    print(f"💡 MAE of {test_mae:.3f} means predictions are typically within ±{test_mae:.3f} points")
print("="*70)

# ============================================================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE RANKINGS")
print("="*70)

# Get feature importance from Random Forest
importance_df = pd.DataFrame({
    'Feature': best_12_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
    bar_length = int(row['Importance'] * 50)
    bar = '█' * bar_length
    print(f"{i:2d}. {row['Feature']:30s} {'|' + bar:52s} {row['Importance']:.4f}")

# ============================================================================
# STEP 10: PREDICTION EXAMPLES
# ============================================================================
print("\n" + "="*70)
print("SAMPLE PREDICTIONS vs ACTUAL SCORES")
print("="*70)

sample_size = min(15, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

comparison_df = pd.DataFrame({
    'Actual Score': y_test.values[sample_indices],
    'Predicted': y_pred_test[sample_indices],
    'Error': np.abs(y_test.values[sample_indices] - y_pred_test[sample_indices])
}).round(3)

comparison_df = comparison_df.sort_values('Actual Score', ascending=False)
print(comparison_df.to_string(index=False))

print(f"\n✅ Average prediction error: {test_mae:.3f} IMDB points")

# ============================================================================
# STEP 11: MODEL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)
print(f"✅ Features used: {len(best_12_features)} (all with zero nulls)")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")
print(f"✅ Model type: Ensemble (Random Forest + Gradient Boosting)")
print(f"✅ Test accuracy: {test_r2*100:.2f}%")
print(f"✅ Prediction error: ±{test_mae:.3f} points")
print("="*70)

print("\n🎉 Model training complete! Your model is ready to predict IMDB scores.")
