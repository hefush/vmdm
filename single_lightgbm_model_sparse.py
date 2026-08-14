import time
import os
import sys
import resource
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, precision_recall_curve, auc
from scipy.sparse import csr_matrix

if len(sys.argv) < 4:
    print('Usage: python %s <train.data.xls> <test.data.xls> <test.predict.xls> [min_ppv:0.90]'%sys.argv[0])
    exit()

trainfile = sys.argv[1]
validfile = sys.argv[2]
outfile = sys.argv[3]

# Adjust threshold based on model performance on test set, requiring PPV >= 0.90.
min_ppv = float(sys.argv[4]) if len(sys.argv) > 4 else 0.90
threshold = 0.5


def get_model_threads():
    try:
        return max(1, int(os.environ.get('VMDM_MODEL_THREADS', os.environ.get('OMP_NUM_THREADS', '2'))))
    except ValueError:
        return 2


model_threads = get_model_threads()

# For recording resource consumption:
start_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime
start_time = time.time()

# Read training data:
df = pd.read_table(trainfile, delimiter='\t')
y = df['Drug']
X = df.drop(['Name', 'Drug'], axis=1)

# Save feature names
feature_names = X.columns

# Convert feature data to float type
X = X.astype(np.float32)

if X.shape[1] == 0:
    raise SystemExit(
        "No overlapping features were found between the query sample and the "
        "training database. Check the VCF, feature list, --min_cov and --max_snps settings."
    )

# Convert to sparse matrix to save memory:
X = csr_matrix(X)

# Split training set
stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# Convert sparse matrix to LightGBM compatible format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Define LightGBM model
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'data_random_seed': 42,
        'deterministic': True,
        'num_threads': model_threads,
}

# Train model
model = lgb.train(params, train_data, num_boost_round=100)

# Predict probabilities
y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_prob >= threshold).astype(int)
# Generate threshold curve:
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
# Calculate precision (PPV):
raw_precision = precision_score(y_test, y_pred, zero_division=0)
# Adjust threshold based on PPV on test set:
if raw_precision < min_ppv:
    eligible = np.where(precision[:-1] >= min_ppv)[0]
    if len(eligible) > 0:
        threshold = max(threshold, thresholds[eligible[0]])
    elif len(thresholds) > 0:
        threshold = max(threshold, thresholds[-1])
    # Make predictions based on new threshold:
    y_pred = (y_pred_prob >= threshold).astype(int)

# Calculate performance scores
model_auc = auc(recall, precision)
model_recall = recall_score(y_test, y_pred, zero_division=0)
model_precision = precision_score(y_test, y_pred, zero_division=0)
model_accuracy = accuracy_score(y_test, y_pred)

end_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime
end_time = time.time()

# Calculate memory peak
mem_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Read test set:
df = pd.read_table(validfile, delimiter='\t')
X_valid = df.drop(['Name'], axis=1)

# Convert feature data to float type
X_valid = X_valid.astype(np.float32)

missing_features = [feature for feature in feature_names if feature not in X_valid.columns]
if missing_features:
    raise SystemExit(
        "The query feature file does not contain all training features: "
        + ", ".join(missing_features[:10])
    )
X_valid = X_valid.loc[:, feature_names]

# Convert to sparse matrix to save memory:
X_valid = csr_matrix(X_valid)

# Predict probabilities
y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)
y_pred = np.where(y_pred_prob >= threshold, 1, 0)

df_out = pd.DataFrame()
df_out['Name'] = df['Name']
df_out['y_pred_prob'] = y_pred_prob
df_out['y_pred'] = y_pred
df_out['model_recall'] = model_recall
df_out['model_precision'] = model_precision
df_out['model_accuracy'] = model_accuracy
df_out['model_auc'] = model_auc
df_out['time'] = end_time - start_time
df_out['cpu'] = end_cpu - start_cpu
df_out['memory_peak'] = mem_peak
df_out['threshold'] = threshold
df_out['raw_precision'] = raw_precision

# Output results:
df_out.to_csv(outfile, sep="\t", index=False)

# Get feature importance
feature_importance = model.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Output feature importance
imp_file = outfile + '.feature.xls'
feature_importance_df.to_csv(imp_file, sep="\t", index=False)
