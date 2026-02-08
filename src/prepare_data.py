import numpy as np
import os

data_path = r'C:\Users\aadir\Project_Instrument\data\openmic-2018.npz'
output_dir = r'C:\Users\aadir\Project_Instrument\data\processed'

def prepdata():
    print(f"Loading dataset from {data_path}")
    try:
        data = np.load(data_path, allow_pickle = True)
    except FileNotFoundError:
        print("File not found")
        return
    
    X_raw = data['X']
    Y_true = data['Y_true']
    Y_mask = data['Y_mask']
    X_averaged = np.mean(X_raw, axis = 1)

    X_averaged = X_averaged.astype(np.float32) / 255.0

    indices = np.arange(len(X_averaged))
    np.random.shuffle(indices)

    split_point = int(0.8 * len(X_averaged))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    X_train = X_averaged[train_idx]
    Y_train = Y_true[train_idx]
    M_train = Y_mask[train_idx]

    X_test = X_averaged[test_idx]
    Y_test = Y_true[test_idx]
    M_test = Y_mask[test_idx]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.savez(f'{output_dir}/train.npz', X = X_train, Y = Y_train, M = M_train)
    np.savez(f'{output_dir}/test.npz', X = X_test, Y = Y_test, M = M_test)

    print(f"Data processed and saved to {output_dir}")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Test Samples: {X_test.shape[0]}")

if __name__ == "__main__":
    prepdata()