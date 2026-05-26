from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('./data')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
TRAIN_LABELS_PATH = DATA_DIR / 'train_labels.csv'
SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

train_files = sorted(TRAIN_DIR.glob('TRAIN_*.csv'))
test_files = sorted(TEST_DIR.glob('TEST_*.csv'))

print(f'train files: {len(train_files)}')
print(f'test files: {len(test_files)}')
train_labels.head()

R_HIT = 0.01

def constant_velocity_predict(sample_path: Path):
    df = pd.read_csv(sample_path)
    prev_xyz = df.loc[df.index[-2], ['x', 'y', 'z']].to_numpy(dtype=float)
    last_xyz = df.loc[df.index[-1], ['x', 'y', 'z']].to_numpy(dtype=float)
    pred_xyz = last_xyz + 2.0 * (last_xyz - prev_xyz)
    return pred_xyz

def build_prediction_df(sample_files):
    rows = []
    for sample_path in sample_files:
        pred_xyz = constant_velocity_predict(sample_path)
        rows.append({
            "id": sample_path.stem,
            "x": pred_xyz[0],
            "y": pred_xyz[1],
            "z": pred_xyz[2],
        })
    return pd.DataFrame(rows)

train_pred = build_prediction_df(train_files)
train_eval = train_labels.merge(train_pred, on='id', suffixes=('_true', '_pred'))

true_xyz = train_eval[['x_true', 'y_true', 'z_true']].to_numpy()
pred_xyz = train_eval[['x_pred', 'y_pred', 'z_pred']].to_numpy()
distance = np.linalg.norm(true_xyz - pred_xyz, axis=1)
hit_rate = np.mean(distance <= R_HIT)

print(f'Constant Velocity Train Hit Rate @ {R_HIT:.3f}m: {hit_rate:.4f}')
print(f'Mean Distance Error: {distance.mean():.6f} m')

test_pred = build_prediction_df(test_files)
test_pred.head()

submission = sample_submission[['id']].merge(test_pred, on='id', how='left')
submission.to_csv('./result/basecode.csv', index=False)
print('saved: basecode.csv')
submission.head()