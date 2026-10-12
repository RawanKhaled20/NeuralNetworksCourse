import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from skimage.feature import hog
from sklearn.svm import SVC
import matlab.engine  # <--- إضافة سطر واحد فقط

# ============== REPRODUCIBILITY SETUP ==============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")

# ============== CONFIG ==============
DATASET_PATH = "Indian_Digits_Train"
IMG_SIZE = 28
GLOBAL_SEED = 42
NUM_SEEDS = 300              # Pipeline 2: 300 seed images
BOUNDARY_PER_ITER = 20       # Active refinement: 20 per iteration
PSEUDO_PER_CLASS = 50        # Self-training: 50 per class
MAX_ITERATIONS = 10

# ============== START MATLAB ENGINE ==============
print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\smart\Downloads', nargout=0)  # مسار check_accuracy.p
print("MATLAB engine ready!")

# ============== REAL ORACLE (MATLAB) ==============

def check_accuracy_real(predictions):
    """
    Real oracle using MATLAB check_accuracy.p
    """
    preds_matlab = matlab.double(predictions.tolist())
    accuracy = eng.check_accuracy(preds_matlab)
    
    if accuracy < 1:
        accuracy = accuracy * 100
    
    print(f"  [ORACLE] Accuracy: {accuracy:.2f}%")
    return float(accuracy)

# ============== 1. LOAD IMAGES ==============
def load_images_from_folder(folder_path):
    print("Loading images...")
    images = []
    filenames = []
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

    for i, file in enumerate(files):
        if file.endswith(".bmp"):
            if i % 1000 == 0:
                print(f"  Loaded {i}/{len(files)} images")
            path = os.path.join(folder_path, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img.flatten())
            filenames.append(file)
    return np.array(images), filenames

# ============== 2. HOG FEATURE EXTRACTION ==============
def extract_hog_features(images):
    print("Extracting HOG features...")
    features = []
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(images)} images")
        img = img.reshape(IMG_SIZE, IMG_SIZE)
        f = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(f)
    return np.array(features)

# ============== 3. DATA AUGMENTATION ==============
def augment_single_image(img_2d, label):
    """
    Generate augmented versions of a single image.
    Returns list of (hog_features, label, weight)
    """
    augmented = []
    
    # 1. Rotations ±5 degrees
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
        rotated = cv2.warpAffine(img_2d, M, (IMG_SIZE, IMG_SIZE))
        hog_feat = hog(rotated, orientations=9, pixels_per_cell=(4,4),
                       cells_per_block=(2,2), block_norm='L2-Hys')
        augmented.append((hog_feat, label, 1))
    
    # 2. Shifts ±2 pixels in 4 directions
    for dx, dy in [(2,0), (-2,0), (0,2), (0,-2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img_2d, M, (IMG_SIZE, IMG_SIZE))
        hog_feat = hog(shifted, orientations=9, pixels_per_cell=(4,4),
                       cells_per_block=(2,2), block_norm='L2-Hys')
        augmented.append((hog_feat, label, 1))
    
    # 3. Gaussian noise (blur)
    noisy = cv2.GaussianBlur(img_2d, (3, 3), 0.5)
    hog_feat = hog(noisy, orientations=9, pixels_per_cell=(4,4),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    augmented.append((hog_feat, label, 1))
    
    return augmented

# ============== 4. MANUAL LABELING TOOL ==============

def manual_labeling(images, indices, title_prefix="Labeling", filenames=None):
    """
    Manual labeling for a set of indices.
    Returns dict: {index: label}
    """
    labels = {}
    total = len(indices)
    
    print(f"\n--- {title_prefix} PHASE ---")
    print(f"Labeling {total} images (10 seconds per image simulated)")
    
    for i, idx in enumerate(indices):
        img = images[idx].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap='gray')
        
        if filenames is not None:
            real_filename = filenames[idx]
            title_text = f"{title_prefix} | {i+1}/{total} | File: {real_filename} (index: {idx})"
        else:
            title_text = f"{title_prefix} | Image {i+1}/{total} | ID: {idx}"
        
        plt.title(title_text)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
        
        valid_input = False
        while not valid_input:
            ans = input(f"Enter label for {real_filename if filenames else idx} (0-9): ")
            if ans.isdigit() and 0 <= int(ans) <= 9:
                labels[idx] = int(ans)
                valid_input = True
            else:
                print("Invalid. Enter 0-9")
        plt.close()
    
    print(f"   Labeled {len(labels)} images")
    return labels
# ============== 5. TRAIN SVM ==============
def train_svm(X_train, y_train, sample_weights):
    """
    Train SVM with RBF kernel and sample weights.
    """
    print("\nTraining SVM with sample weights...")
    
    svm_model = SVC(
        kernel='rbf',
        decision_function_shape='ovo',
        gamma='scale',
        probability=True,
        random_state=GLOBAL_SEED
    )
    
    svm_model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"SVM training complete! Trained on {len(X_train)} samples")
    
    return svm_model

# ============== 6. GET BOUNDARY SAMPLES (Active Refinement) ==============
def get_boundary_samples(model, features, current_labels, num_samples=20):
    """
    Select ambiguous samples ONLY from UNLABELED data.
    """
    print(f"\nSelecting {num_samples} boundary samples from UNLABELED data...")
    
    # Find unlabeled samples (where label == -1)
    unlabeled_mask = (current_labels == -1)
    unlabeled_indices = np.where(unlabeled_mask)[0]
    
    print(f"  Found {len(unlabeled_indices)} unlabeled samples")
    
    if len(unlabeled_indices) == 0:
        print("  No unlabeled samples left!")
        return np.array([])
    
    # Get decision function scores for unlabeled samples
    unlabeled_features = features[unlabeled_mask]
    scores = model.decision_function(unlabeled_features)
    
    # Calculate margin = difference between top two scores
    n_classes = len(model.classes_)
    
    if n_classes == 2:
        # Binary case
        margin = np.abs(scores.flatten())
    else:
        # Multi-class: use decision_function or probability
        # Simple approach: use probability differences if available
        try:
            probs = model.predict_proba(unlabeled_features)
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        except:
            # Fallback: use score differences
            margin = np.zeros(len(unlabeled_features))
            for i in range(len(scores)):
                sorted_scores = np.sort(scores[i])[::-1]
                margin[i] = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    
    # Get most ambiguous (smallest margin)
    boundary_local = np.argsort(margin)[:num_samples]
    full_indices = unlabeled_indices[boundary_local]
    
    print(f"  Selected {len(full_indices)} boundary samples")
    return full_indices

# ============== 7. GET HIGH CONFIDENCE SAMPLES (Self-Training) ==============
def get_high_confidence_samples(model, features, current_labels, percentile=75, per_class=50):
    """
    Select high-confidence predictions for self-training.
    """
    print(f"\nSelecting high-confidence samples for self-training...")
    
    # Find unlabeled samples
    unlabeled_mask = (current_labels == -1)
    unlabeled_indices = np.where(unlabeled_mask)[0]
    
    if len(unlabeled_indices) == 0:
        print("  No unlabeled samples available")
        return np.array([]), np.array([])
    
    # Get predictions and confidence scores
    unlabeled_features = features[unlabeled_mask]
    predictions = model.predict(unlabeled_features)
    
    # Get confidence scores
    try:
        probs = model.predict_proba(unlabeled_features)
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    except:
        # Fallback
        margins = np.ones(len(unlabeled_features)) * 0.5
    
    # Calculate threshold
    threshold = np.percentile(margins, percentile)
    print(f"  Confidence threshold: {threshold:.3f} (75th percentile)")
    
    selected_indices = []
    selected_predictions = []
    
    for digit in range(10):
        # Find predictions for this digit
        digit_mask = (predictions == digit) & (margins > threshold)
        digit_local_indices = np.where(digit_mask)[0]
        
        if len(digit_local_indices) > 0:
            # Sort by margin (highest confidence first)
            digit_margins = margins[digit_local_indices]
            sorted_local = digit_local_indices[np.argsort(digit_margins)[::-1]]
            top_k = sorted_local[:per_class]
            
            for local_idx in top_k:
                selected_indices.append(unlabeled_indices[local_idx])
                selected_predictions.append(predictions[local_idx])
    
    print(f"  Selected {len(selected_indices)} high-confidence samples")
    return np.array(selected_indices), np.array(selected_predictions)

# ============== MAIN FUNCTION ==============
if __name__ == "__main__":
    # -------------------- 0. Set seed --------------------
    set_seed(GLOBAL_SEED)
    
    # -------------------- 1. Load images --------------------
    print("\n" + "="*60)
    print("PIPELINE 2: Manual Seed Labelling with Augmentation")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n ERROR: Dataset path '{DATASET_PATH}' not found!")
        print("Please update DATASET_PATH variable with correct path.")
        exit(1)
    
    images, filenames = load_images_from_folder(DATASET_PATH)
    print(f"\n Loaded {len(images)} images")
    
    # -------------------- 2. Extract HOG features --------------------
    hog_features = extract_hog_features(images)
    print(f" HOG features shape: {hog_features.shape}")
    
    # -------------------- 3. Initialize label arrays --------------------
    current_labels = np.full(len(images), -1)  # -1 means unlabeled
    
    # -------------------- 4. STEP 1: Create Seed Set (300 images) --------------------
    print("\n" + "-"*40)
    print("STEP 1: Creating Seed Set (300 images)")
    print("-"*40)
    
    seed_file = "pipeline2_seed_labels.pkl"
    
    if os.path.exists(seed_file):
        print(f"Loading existing seed labels from {seed_file}")
        with open(seed_file, 'rb') as f:
            seed_labels = pickle.load(f)
    else:
        # Randomly select 300 images
        all_indices = list(range(len(images)))
        seed_indices = random.sample(all_indices, NUM_SEEDS)
        
        # Manual labeling (10 seconds per image)
        print(f"\n  You will need to label {NUM_SEEDS} images manually.")
        print(f"This will take approximately {NUM_SEEDS * 10 / 60:.1f} minutes.")
        input("Press Enter to continue...")
        
        seed_labels = manual_labeling(images, seed_indices, "SEED SET", filenames)
        
        # Save for future runs
        with open(seed_file, 'wb') as f:
            pickle.dump(seed_labels, f)
    
    # Prepare training data with seed images and augmentations
    X_train_list = []
    y_train_list = []
    weight_list = []
    
    print("\nGenerating augmented versions of seed images...")
    for idx, lbl in seed_labels.items():
        # Original seed image (weight = 100)
        X_train_list.append(hog_features[idx])
        y_train_list.append(lbl)
        weight_list.append(100)
        current_labels[idx] = lbl
        
        # Generate augmented versions (weight = 1)
        img_2d = images[idx].reshape(IMG_SIZE, IMG_SIZE)
        augmented = augment_single_image(img_2d, lbl)
        
        for aug_feat, aug_lbl, aug_weight in augmented:
            X_train_list.append(aug_feat)
            y_train_list.append(aug_lbl)
            weight_list.append(aug_weight)
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    sample_weights = np.array(weight_list)
    
    print(f"\nInitial training set:")
    print(f"  - Original seeds: {len(seed_labels)} (weight=100)")
    print(f"  - Augmented copies: {len(X_train) - len(seed_labels)} (weight=1)")
    print(f"  - Total training samples: {len(X_train)}")
    
    # -------------------- 5. Train initial SVM --------------------
    svm_model = train_svm(X_train, y_train, sample_weights)
    
    # -------------------- 6. Initial evaluation --------------------
    print("\n" + "-"*40)
    print("Initial Evaluation")
    print("-"*40)
    all_preds = svm_model.predict(hog_features)
    accuracy = check_accuracy_real(all_preds)  # <--- Real oracle now!
    
    # -------------------- 7. Iterative Refinement Loop --------------------
    print("\n" + "="*60)
    print("STARTING ITERATIVE REFINEMENT")
    print("="*60)
    
    prev_accuracy = accuracy
    all_human_labels = seed_labels.copy()
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}")
        print(f"{'='*50}")
        
        # Get current predictions for margin calculation
        all_preds = svm_model.predict(hog_features)
        
        # ----- Step 2: Active Refinement (Boundary Images) -----
        print("\n" + "-"*40)
        print("STEP 2: Active Refinement (Boundary Images)")
        print("-"*40)
        
        boundary_indices = get_boundary_samples(svm_model, hog_features, current_labels, BOUNDARY_PER_ITER)
        
        if len(boundary_indices) > 0:
            # Manual labeling for boundary images
            print(f"\n  Labeling {len(boundary_indices)} boundary images...")
            human_labels = manual_labeling(images, boundary_indices, f"Boundary Iter {iteration}", filenames)
            all_human_labels.update(human_labels)
            
            # Update training data with new labels (weight = 100)
            for idx, lbl in human_labels.items():
                X_train = np.vstack([X_train, hog_features[idx]])
                y_train = np.append(y_train, lbl)
                sample_weights = np.append(sample_weights, 100)
                current_labels[idx] = lbl
            
            print(f"   Added {len(human_labels)} boundary images (weight=100)")
        
        # ----- Step 3: Self-Training (High Confidence) -----
        print("\n" + "-"*40)
        print("STEP 3: Self-Training (High Confidence Predictions)")
        print("-"*40)
        
        high_conf_indices, high_conf_preds = get_high_confidence_samples(
            svm_model, hog_features, current_labels,
            percentile=75, per_class=PSEUDO_PER_CLASS
        )
        
        # Add pseudo-labels (weight = 1)
        pseudo_added = 0
        for idx, pred_label in zip(high_conf_indices, high_conf_preds):
            if current_labels[idx] == -1:  # Still unlabeled
                X_train = np.vstack([X_train, hog_features[idx]])
                y_train = np.append(y_train, pred_label)
                sample_weights = np.append(sample_weights, 1)
                current_labels[idx] = pred_label
                pseudo_added += 1
        
        print(f"   Added {pseudo_added} pseudo-labeled images (weight=1)")
        
        # ----- Step 4: Retrain SVM -----
        print("\n" + "-"*40)
        print("STEP 4: Retraining SVM")
        print("-"*40)
        
        svm_model = train_svm(X_train, y_train, sample_weights)
        
        # ----- Step 5: Evaluate -----
        print("\n" + "-"*40)
        print("STEP 5: Evaluation")
        print("-"*40)
        
        all_preds = svm_model.predict(hog_features)
        accuracy = check_accuracy_real(all_preds)  # <--- Real oracle now!
        print(f"\nAccuracy after iteration {iteration}: {accuracy:.2f}%")
        print(f"Improvement: {accuracy - prev_accuracy:.2f}%")
        
        # ----- Step 6: Stopping Criteria -----
        if accuracy >= 99.0:
            print(f"\n TARGET REACHED! {accuracy:.2f}% >= 99%")
            break
        
        if iteration > 1 and (accuracy - prev_accuracy) < 0.1:
            print(f"\n CONVERGENCE: Improvement {accuracy-prev_accuracy:.2f}% < 0.1%")
            break
        
        prev_accuracy = accuracy
    
    # -------------------- 8. FINAL REPORT --------------------
    print("\n" + "="*60)
    print("                 PIPELINE 2 FINAL REPORT")
    print("="*60)
    
    total_manual = len(all_human_labels)
    total_time_sec = total_manual * 10
    total_time_min = total_time_sec / 60
    pseudo_count = np.sum((current_labels != -1)) - total_manual
    
    print(f"{'Metric':<35} {'Value':>25}")
    print("-"*60)
    print(f"{'Final Accuracy':<35} {accuracy:>24.2f}%")
    print(f"{'Total Iterations':<35} {iteration:>25}")
    print(f"{'Seed Set Size':<35} {NUM_SEEDS:>25}")
    print(f"{'Boundary Images Total':<35} {total_manual - NUM_SEEDS:>25}")
    print(f"{'Total Manual Images':<35} {total_manual:>25}")
    print(f"{'Manual Time (seconds)':<35} {total_time_sec:>25}")
    print(f"{'Manual Time (minutes)':<35} {total_time_min:>25.1f}")
    print(f"{'Pseudo-labeled Images':<35} {pseudo_count:>25}")
    print(f"{'Total Training Samples':<35} {len(X_train):>25}")
    print(f"{'Human Label Weight':<35} {'100':>25}")
    print(f"{'Augmented/Pseudo Weight':<35} {'1':>25}")
    print(f"{'SVM Kernel':<35} {'RBF (Radial Basis Function)':>25}")
    print(f"{'Stopping Condition':<35} {('99% reached' if accuracy>=99 else 'Convergence'):>25}")
    print("="*60)
    
    # Save final predictions
    np.savetxt('pipeline2_final_predictions.txt', all_preds, fmt='%d')
    np.save('pipeline2_final_labels.npy', current_labels)
    
    print("\n Final predictions saved to:")
    print("   - pipeline2_final_predictions.txt")
    print("   - pipeline2_final_labels.npy")
    
    # Close MATLAB engine
    eng.quit()
    print("\n MATLAB engine closed")

