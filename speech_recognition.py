import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from PIL import Image
import io
import time
from collections import Counter


# ── Fix random seed for reproducibility ─────────────────────
# This ensures same random initialization every run
# so all 4 parts start from the same conditions (fair comparison)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# FUNCTION: Load all wav files from a folder
def load_speech_data(folder_path):
    """
    Loads all .wav files from a folder.
    Extracts label from filename: e.g. C04n_0.wav → label = 0

    Returns:
        data: list of (audio_signal, sample_rate, label)
    """
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):

            # Extract label from filename (number after last underscore)
            name_without_ext = filename.replace(".wav", "")   # C04n_0
            label = int(name_without_ext.split("_")[-1])      # 0

            # Full path to the file
            file_path = os.path.join(folder_path, filename)

            # Load audio
            audio, sr = librosa.load(file_path, sr=None)   # sr=None keeps original sample rate

            data.append((audio, sr, label))

    return data


# FUNCTION: Convert audio → Mel Spectrogram
def audio_to_spectrogram(audio, sr, n_mels=28, image_size=(28,28)):
    """
    Converts raw audio signal to a Mel Spectrogram image.

    Args:
        audio      : raw audio signal (numpy array)
        sr         : sample rate (e.g. 16000)
        n_mels     : number of mel frequency bands (height of image)
        image_size : final (height, width) to resize the spectrogram

    Returns:
        spec_normalized: 2D numpy array of shape (28,28), values in [0, 1]
    """
    # Step 1: Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)

    # Step 2: Convert power to decibels (log scale) — like human hearing
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Step 3: Resize to fixed size using simple interpolation
    from PIL import Image
    img = Image.fromarray(mel_spec_db)
    img_resized = img.resize((image_size[1], image_size[0]))   # (width, height)
    spec_array = np.array(img_resized)

    # Step 4: Normalize to [0, 1]
    spec_min = spec_array.min()
    spec_max = spec_array.max()
    spec_normalized = (spec_array - spec_min) / (spec_max - spec_min + 1e-6)

    return spec_normalized  # shape: (28,28)

# FUNCTION: Convert full dataset to spectrograms
def dataset_to_spectrograms(data, n_mels=28, image_size=(28,28)):
    """
    Converts a full dataset of audio files to spectrograms.

    Args:
        data: list of (audio, sr, label)  ← output of load_speech_data()

    Returns:
        X: numpy array of shape (N, 28,28)
        y: numpy array of shape (N,)
    """
    X = []
    y = []

    for i, (audio, sr, label) in enumerate(data):
        spec = audio_to_spectrogram(audio, sr, n_mels=n_mels, image_size=image_size)
        X.append(spec)
        y.append(label)

        # Progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data)} files...")

    X = np.array(X)   # shape: (N, 28,28)
    y = np.array(y)   # shape: (N,)

    return X, y

# CNN Model Definition
class SpectrogramCNN(nn.Module):
    """
    Exact LeNet-5 Architecture adapted for speech spectrogram recognition.
    Matches the structure shown in the assignment PDF.

    Input  : (batch, 1, 28, 28)  
    Conv1  : 6 filters, 5×5  → BatchNorm → ReLU → AvgPool  → (batch, 6,  14, 14)  (was 28×28×6 → 14×14×6)
    Conv2  : 16 filters, 5×5 → BatchNorm → ReLU → AvgPool  → (batch, 16,  5,  5)  (was 10×10×16 → 5×5×16)
    Flatten: 16×5×5 = 400
    FC1    : 400 → 120 → BatchNorm → ReLU → Dropout
    FC2    : 120 → 84  → BatchNorm → ReLU → Dropout
    FC3    : 84  → 10  ← one score per digit class
    Output : (batch, 10)
    """
    def __init__(self):
        super(SpectrogramCNN, self).__init__()

        # Conv Block 1: 1→6 filters, 5×5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)       # 28×28 → 24×24
        self.bn1   = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2)          # 24×24 → 12×12

        # Conv Block 2: 6→16 filters, 5×5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)      # 12×12 → 8×8
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2)          # 8×8   → 4×4

        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1     = nn.Linear(16 * 4 * 4, 120)        # 256 → 120
        self.bn3     = nn.BatchNorm1d(120)
        self.dropout1= nn.Dropout(0.25)

        self.fc2     = nn.Linear(120, 84)
        self.bn4     = nn.BatchNorm1d(84)
        self.dropout2= nn.Dropout(0.25)

        self.fc3     = nn.Linear(84, 10)                  # 10 digit classes

    def forward(self, x):
        # Block 1
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        # Flatten
        x = self.flatten(x)
        # FC layers
        x = self.dropout1(self.relu(self.bn3(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn4(self.fc2(x))))
        x = self.fc3(x)
        return x

# FUNCTION: Custom PyTorch Dataset
class SpectrogramDataset(Dataset):
    """
    Wraps X (spectrograms) and y (labels) into a PyTorch Dataset.
    Input X shape: (N, 28,28) → will add channel dim → (N, 1, 28,28)
    """
    def __init__(self, X, y):
        # Add channel dimension: (N, 28,28) → (N, 1, 28,28)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# FUNCTION: Train the model for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track stats
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy



# FUNCTION: Evaluate the model on test set
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# FUNCTION: Full training loop
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=50, batch_size=28, lr=0.0001):
    """
    Full training loop.
    Returns: trained model + history dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model = model.to(device)

    # Create datasets and loaders
    train_dataset = SpectrogramDataset(X_train, y_train)
    test_dataset  = SpectrogramDataset(X_test,  y_test)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009, weight_decay=0.001)
    # History to plot later
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print(f"\n{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Test Loss':<14} {'Test Acc'}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"{epoch:<8} {train_loss:<14.4f} {train_acc:<14.2f} {test_loss:<14.4f} {test_acc:.2f}%")

    print(f"\n Training complete!")
    print(f"   Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   Final Test  Accuracy: {history['test_acc'][-1]:.2f}%")

    return model, history

# FUNCTION: Speech Augmentation
def augment_speech(audio, sr):
    """
    Takes one audio signal and returns 3 augmented versions:
        1. Speed up 3%
        2. Speed down 3%
        3. Original + random noise

    Args:
        audio : raw audio signal (numpy array)
        sr    : sample rate

    Returns:
        list of 3 augmented audio signals
    """
    augmented = []

    # 1. Speed up 3%
    audio_fast = librosa.effects.time_stretch(audio, rate=1.03)
    augmented.append(audio_fast)

    # 2. Speed down 3%
    audio_slow = librosa.effects.time_stretch(audio, rate=0.97)
    augmented.append(audio_slow)

    # 3. Add random noise
    noise = np.random.randn(len(audio))
    audio_noisy = audio + 0.01 * noise
    augmented.append(audio_noisy)

    return augmented  # list of 3 audio arrays



# FUNCTION: Build augmented dataset (speech level)  — Part (b)
def build_speech_augmented_dataset(train_data, n_mels=28, image_size=(28,28)):
    """
    Takes original train_data and adds augmented speech versions.
    Original 1200 + 3 augmented each = 4800 total samples.

    Args:
        train_data: list of (audio, sr, label)  ← from load_speech_data()

    Returns:
        X: numpy array (N, 28,28)
        y: numpy array (N,)
    """
    X = []
    y = []

    for i, (audio, sr, label) in enumerate(train_data):

        # Original spectrogram
        spec = audio_to_spectrogram(audio, sr, n_mels=n_mels, image_size=image_size)
        X.append(spec)
        y.append(label)

        # Augmented versions
        augmented_audios = augment_speech(audio, sr)
        for aug_audio in augmented_audios:
            spec_aug = audio_to_spectrogram(aug_audio, sr, n_mels=n_mels, image_size=image_size)
            X.append(spec_aug)
            y.append(label)   # same label!

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(train_data)} files...")

    X = np.array(X)
    y = np.array(y)
    return X, y 

# FUNCTION: Image (spectrogram) Augmentation
def augment_spectrogram(spec):
    """
    Takes one spectrogram image (28x28) and returns 3 augmented versions:
        1. Squeeze horizontally by 3%  (compress time axis)
        2. Expand horizontally by 3%   (stretch time axis)
        3. Add random noise to image

    Args:
        spec : 2D numpy array (28,28), values in [0,1]

    Returns:
        list of 3 augmented spectrogram arrays, each (28,28)
    """
    from PIL import Image

    augmented = []
    h, w = spec.shape   # 28,28

    # Convert to PIL image for resizing
    img = Image.fromarray((spec * 255).astype(np.uint8))

    # 1. Squeeze horizontally by 3% → resize to (62, 28) → pad back to (28,28)
    new_w_squeeze = int(w * 0.97)
    img_squeezed = img.resize((new_w_squeeze, h))
    arr_squeezed = np.array(img_squeezed) / 255.0
    # Pad right side with zeros to get back to 28x28
    pad_width = w - new_w_squeeze
    arr_squeezed = np.pad(arr_squeezed, ((0, 0), (0, pad_width)), mode='constant')
    augmented.append(arr_squeezed)

    # 2. Expand horizontally by 3% → resize to (66, 28) → crop back to (28,28)
    new_w_expand = int(w * 1.03)
    img_expanded = img.resize((new_w_expand, h))
    arr_expanded = np.array(img_expanded) / 255.0
    # Crop to 28x28
    arr_expanded = arr_expanded[:, :w]
    augmented.append(arr_expanded)

    # 3. Add random noise to image
    noise = np.random.randn(h, w) * 0.05
    arr_noisy = np.clip(spec + noise, 0, 1)
    augmented.append(arr_noisy)

    return augmented   # list of 3 arrays, each (28,28)

# FUNCTION: Build augmented dataset (image level) — Part (c)
def build_image_augmented_dataset(train_data, n_mels=28, image_size=(28,28)):
    """
    Converts audio to spectrograms first, then augments the images.
    Original 1200 → spectrograms → augment images → 4800 total

    Args:
        train_data: list of (audio, sr, label)

    Returns:
        X: numpy array (4800, 28,28)
        y: numpy array (4800,)
    """
    X = []
    y = []

    for i, (audio, sr, label) in enumerate(train_data):

        # Convert audio to spectrogram first
        spec = audio_to_spectrogram(audio, sr, n_mels=n_mels, image_size=image_size)

        # Keep original spectrogram
        X.append(spec)
        y.append(label)

        # Augment the spectrogram image
        augmented_specs = augment_spectrogram(spec)
        for aug_spec in augmented_specs:
            X.append(aug_spec)
            y.append(label)   # same label!

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(train_data)} files...")

    X = np.array(X)
    y = np.array(y)
    return X, y



# FUNCTION: Build augmented dataset (speech + image) — Part (d)
def build_combined_augmented_dataset(train_data, n_mels=28, image_size=(28,28)):
    """
    Applies BOTH speech augmentation AND image augmentation.
    1200 audios
        → speech augment → 4800 audios
        → convert to spectrograms → 4800 images
        → image augment → 4800 × 4 = 19200 images  (original + 3 image versions each)

    Args:
        train_data: list of (audio, sr, label)

    Returns:
        X: numpy array (19200, 28,28)
        y: numpy array (19200,)
    """
    X = []
    y = []

    for i, (audio, sr, label) in enumerate(train_data):

        # Get original + 3 speech augmented versions
        all_audios = [audio] + augment_speech(audio, sr)   # 4 audio versions

        for aug_audio in all_audios:
            # Convert each audio to spectrogram
            spec = audio_to_spectrogram(aug_audio, sr, n_mels=n_mels, image_size=image_size)

            # Keep original spectrogram
            X.append(spec)
            y.append(label)

            # Augment spectrogram image → 3 more versions
            augmented_specs = augment_spectrogram(spec)
            for aug_spec in augmented_specs:
                X.append(aug_spec)
                y.append(label)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(train_data)} files...")

    X = np.array(X)
    y = np.array(y)
    return X, y





# TEST / MAIN — Run this to check the function

if __name__ == "__main__":

    TRAIN_FOLDER = "C:\\Users\\smart\\Desktop\\mypython\\audio\\audio-dataset\\Train"
    TEST_FOLDER  = "C:\\Users\\smart\\Desktop\\mypython\\audio\\audio-dataset\\Test"

    print("Loading training data...")
    train_data = load_speech_data(TRAIN_FOLDER)
    print(f"   Loaded {len(train_data)} training files")

    print("Loading test data...")
    test_data = load_speech_data(TEST_FOLDER)
    print(f"   Loaded {len(test_data)} test files")

    # Show a few samples
    print("\n--- Sample check (first 5 files) ---")
    for i, (audio, sr, label) in enumerate(train_data[:5]):
        print(f"  File {i+1}: label={label}, sample_rate={sr}, audio_length={len(audio)} samples, duration={len(audio)/sr:.2f} sec")

    # Check label distribution
    labels = [label for (_, _, label) in train_data]
    print("\n--- Label distribution in training set ---")
    for digit, count in sorted(Counter(labels).items()):
        print(f"  Digit {digit}: {count} files")

    # Convert to spectrograms
    print("\nConverting training data to spectrograms...")
    X_train, y_train = dataset_to_spectrograms(train_data)

    print("\nConverting test data to spectrograms...")
    X_test, y_test = dataset_to_spectrograms(test_data)

    # Shape check
    print(f"\n X_train shape: {X_train.shape}")   # should be (1200, 28,28)
    print(f" y_train shape: {y_train.shape}")     # should be (1200,)
    print(f" X_test shape:  {X_test.shape}")      # should be (300, 28,28)
    print(f" y_test shape:  {y_test.shape}")      # should be (300,)

    # --- Visualize one spectrogram per digit ---
    print("\nShowing one spectrogram per digit...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Mel Spectrogram — One Example Per Digit", fontsize=14)

    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]   # first sample of this digit
        ax = axes[digit // 5][digit % 5]
        ax.imshow(X_train[idx], aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Digit: {digit}")
        ax.axis('off')

    plt.tight_layout()
    #plt.show()

    # PART (A) — Build model No AUGMENTED training set

    model = SpectrogramCNN()
    # Step 4: Train
    print("\nStarting training (part a - no augmentation)...")
    start = time.time()
    model, history = train_model(model, X_train, y_train, X_test, y_test,
                                 epochs=20, batch_size=56, lr=0.0009)
    end = time.time()
    print(f"   Training time: {(end - start)*1000:.1f} msec")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_acc"], label="Train")
    ax1.plot(history["test_acc"],  label="Test")
    ax1.set_title("Accuracy per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    ax2.plot(history["train_loss"], label="Train")
    ax2.plot(history["test_loss"],  label="Test")
    ax2.set_title("Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()     
    #plt.show()


    # PART (B) — Build AUGMENTED training set (speech level)
    print("\nBuilding speech-augmented training set...")
    X_train_aug, y_train_aug = build_speech_augmented_dataset(train_data)

    # Check sizes
    print(f"\n Augmented X_train shape: {X_train_aug.shape}")  # should be (4800, 28,28)
    print(f" Augmented y_train shape: {y_train_aug.shape}")
    print(f" X_test shape:            {X_test.shape}")         # stays (300, 28,28)

    # Check label distribution still balanced
    print("\n--- Label distribution after augmentation ---")
    for digit, count in sorted(Counter(y_train_aug.tolist()).items()):
        print(f"  Digit {digit}: {count} samples")

    # Train model on augmented data
    print("\nStarting training (part b - speech augmentation)...")
    model_b = SpectrogramCNN()
    start = time.time()
    model_b, history_b = train_model(model_b, X_train_aug, y_train_aug,
                                     X_test, y_test,
                                     epochs=50, batch_size=28, lr=0.0009)
    end = time.time()
    print(f"   Training time: {(end-start)*1000:.1f} msec")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Part (b) — Speech Augmentation", fontsize=13)

    ax1.plot(history_b["train_acc"], label="Train")
    ax1.plot(history_b["test_acc"],  label="Test")
    ax1.set_title("Accuracy per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    ax2.plot(history_b["train_loss"], label="Train")
    ax2.plot(history_b["test_loss"],  label="Test")
    ax2.set_title("Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()        
    #plt.show()

    
    # PART (c) — Image Augmentation
    
    print("\nBuilding image-augmented training set...")
    X_train_img, y_train_img = build_image_augmented_dataset(train_data)
    print(f" Image-augmented X_train shape: {X_train_img.shape}")  # (4800, 28,28)

    print("\nStarting training (part c - image augmentation)...")
    model_c = SpectrogramCNN()
    start = time.time()
    model_c, history_c = train_model(model_c, X_train_img, y_train_img,
                                     X_test, y_test,
                                     epochs=50, batch_size=28, lr=0.0009)
    end = time.time()
    print(f"   Training time: {(end-start)*1000:.1f} msec")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Part (c) — Image Augmentation", fontsize=13)
    ax1.plot(history_c["train_acc"], label="Train")
    ax1.plot(history_c["test_acc"],  label="Test")
    ax1.set_title("Accuracy per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax2.plot(history_c["train_loss"], label="Train")
    ax2.plot(history_c["test_loss"],  label="Test")
    ax2.set_title("Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()     
    #plt.show()
    
    # PART (d) — Speech + Image Augmentation
    
    print("\nBuilding combined (speech+image) augmented training set...")
    X_train_comb, y_train_comb = build_combined_augmented_dataset(train_data)
    print(f" Combined augmented X_train shape: {X_train_comb.shape}")  # (19200, 28,28)

    print("\nStarting training (part d - speech + image augmentation)...")
    model_d = SpectrogramCNN()
    start = time.time()
    model_d, history_d = train_model(model_d, X_train_comb, y_train_comb,
                                     X_test, y_test,
                                     epochs=50, batch_size=28, lr=0.001)
    end = time.time()
    print(f"   Training time: {(end-start)*1000:.1f} msec")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Part (d) — Speech + Image Augmentation", fontsize=13)
    ax1.plot(history_d["train_acc"], label="Train")
    ax1.plot(history_d["test_acc"],  label="Test")
    ax1.set_title("Accuracy per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax2.plot(history_d["train_loss"], label="Train")
    ax2.plot(history_d["test_loss"],  label="Test")
    ax2.set_title("Loss per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()     
    #plt.show()

    
    # FINAL SUMMARY — Compare all 4 parts
    
    print("\n" + "="*50)
    print("        FINAL RESULTS SUMMARY for last epoch (may not be best epoch)")
    print("="*50)
    print(f"  Part (a) No augmentation:          {history['test_acc'][-1]:.2f}%")
    print(f"  Part (b) Speech augmentation:      {history_b['test_acc'][-1]:.2f}%")
    print(f"  Part (c) Image augmentation:       {history_c['test_acc'][-1]:.2f}%")
    print(f"  Part (d) Speech + Image aug:       {history_d['test_acc'][-1]:.2f}%")
    print("="*50)    

    print("\n" + "="*50)
    print("        FINAL RESULTS SUMMARY for BEST EPOCH")
    print("="*50)
    print(f"  Part (a) No augmentation:          {max(history['test_acc']):.2f}%")
    print(f"  Part (b) Speech augmentation:      {max(history_b['test_acc']):.2f}%")
    print(f"  Part (c) Image augmentation:       {max(history_c['test_acc']):.2f}%")
    print(f"  Part (d) Speech + Image aug:       {max(history_d['test_acc']):.2f}%")
    print("="*50)  
       
       