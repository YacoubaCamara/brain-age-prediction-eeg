import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print('All libraries loaded successfully.')

DATA_DIR = './data'

def load_eeg_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    match = re.search(r'Age\s*=\s*(\d+)', first_line)
    if not match:
        return None, None
    age = int(match.group(1))

    signal_df = pd.read_csv(filepath, skiprows=1)
    return age, signal_df


ages = []
signals = []
filenames = []

csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

for fname in csv_files:
    fpath = os.path.join(DATA_DIR, fname)
    age, signal_df = load_eeg_file(fpath)
    if age is not None and signal_df is not None:
        ages.append(age)
        signals.append(signal_df)
        filenames.append(fname)

print(f'Loaded {len(ages)} files.')
print(f'Age range: {min(ages)} - {max(ages)} years')
print(f'Mean age: {np.mean(ages):.1f}, Std: {np.std(ages):.1f}')

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

SAMPLING_RATE = 256

def compute_band_power(signal, fs, band):
    freqs, psd = welch(signal, fs=fs, nperseg=min(fs * 2, len(signal)))
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[idx])

def extract_features(signal_df, fs=SAMPLING_RATE):
    numeric_cols = signal_df.select_dtypes(include=[np.number]).columns
    data = signal_df[numeric_cols].values

    abs_powers = {band: [] for band in BANDS}

    for ch_idx in range(data.shape[1]):
        channel_signal = data[:, ch_idx]
        for band_name, band_range in BANDS.items():
            power = compute_band_power(channel_signal, fs, band_range)
            abs_powers[band_name].append(power)

    avg_abs = {band: np.mean(powers) for band, powers in abs_powers.items()}
    total_power = sum(avg_abs.values())
    avg_rel = {band: avg_abs[band] / total_power for band in BANDS}

    feature_names = [f'abs_{b}' for b in BANDS] + [f'rel_{b}' for b in BANDS]
    feature_values = [avg_abs[b] for b in BANDS] + [avg_rel[b] for b in BANDS]

    return feature_values, feature_names


print('Extracting features...')
all_features = []
feature_names = None

for i, signal_df in enumerate(signals):
    features, names = extract_features(signal_df)
    all_features.append(features)
    if feature_names is None:
        feature_names = names
    if (i + 1) % 20 == 0:
        print(f'  Processed {i + 1}/{len(signals)} files...')

X = np.array(all_features)
y = np.array(ages)

print(f'Done. Feature matrix shape: {X.shape}')
print(f'Features: {feature_names}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost':           XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'SVM':               SVR(kernel='rbf', C=10, gamma='scale'),
    'KNN':               KNeighborsRegressor(n_neighbors=5)
}

loo = LeaveOneOut()
results = {}

print('Training models...')
print(f'{"Model":<20} {"MAE (years)":>12} {"R2 Score":>10}')
print('-' * 45)

for name, model in models.items():
    y_pred_all = []
    y_true_all = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred_all.append(model.predict(X_test)[0])
        y_true_all.append(y_test[0])

    mae = mean_absolute_error(y_true_all, y_pred_all)
    r2  = r2_score(y_true_all, y_pred_all)

    results[name] = {
        'MAE': mae,
        'R2': r2,
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }

    print(f'{name:<20} {mae:>12.2f} {r2:>10.3f}')

print('\nDone.')

# Plot 1: Model Comparison
model_names = list(results.keys())
maes = [results[m]['MAE'] for m in model_names]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(model_names, maes, color='steelblue', edgecolor='white')
ax.set_xlabel('Mean Absolute Error (years)')
ax.set_title('Model Comparison — Brain Age Prediction MAE')
ax.invert_yaxis()

for bar, mae in zip(bars, maes):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{mae:.1f} yrs', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print('Saved: model_comparison.png')

# Plot 2: Brain Age Gap
best_model_name = min(results, key=lambda m: results[m]['MAE'])
best = results[best_model_name]

y_true = np.array(best['y_true'])
y_pred = np.array(best['y_pred'])
brain_age_gap = y_pred - y_true

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(y_true, y_pred, alpha=0.7, color='steelblue', edgecolors='white', s=60)
lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Age (years)')
ax.set_ylabel('Predicted Age (years)')
ax.set_title(f'Predicted vs Actual Age — {best_model_name}')
ax.legend()

ax2 = axes[1]
ax2.hist(brain_age_gap, bins=15, color='steelblue', edgecolor='white')
ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No gap')
ax2.set_xlabel('Brain Age Gap (years)')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Brain Age Gap')
ax2.legend()

plt.tight_layout()
plt.savefig('brain_age_gap.png', dpi=150)
plt.show()

print(f'Best model: {best_model_name}')
print(f'MAE: {best["MAE"]:.2f} years')
print(f'R2:  {best["R2"]:.3f}')

# Plot 3: Feature Importance
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_scaled, y)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df['Feature'], importance_df['Importance'],
        color='steelblue', edgecolor='white')
ax.set_xlabel('Feature Importance')
ax.set_title('EEG Band Power Feature Importance (Gradient Boosting)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print('Saved: feature_importance.png')