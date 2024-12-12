#Extracts data from the .mat files that is relavent to the neural network and saves it as a numpy file. This was part of my data preparation
import os
import scipy.io
import numpy as np

mat_data = scipy.io.loadmat(#Omited for privacy reasons)

Output20220811 = mat_data['output']
Wavelengths = #Omited for privacy reasons[0][0][5]

save_dir = #Omited for privacy reasons

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#iterates
for sat_index in range(20):
    # Extract Norm and UTC data for the current satellite
    AMC_15_20220813_Norm = #Omited for privacy reasons[0][sat_index][3]
    AMC_15_20220813_UTC = #Omited for privacy reasons[0][sat_index][6]
    
    # Sat name extracted
    sat_name = #Omited for privacy reasons[0][sat_index][0]

    # Nested arrays trimmed to only show the date and times
    trimmed_utc = []
    for item in #Omited for privacy reasons:
        trimmed_utc.append(item[0][0][0])
    
    # Data to be saved as a Numpy array
    data_to_save = {
        'Norm': #Omited for privacy reasons,
        'Trimmed_UTC': np.array(trimmed_utc),
        'Wavelengths': Wavelengths
    }
    
    
    np.save(os.path.join(save_dir, f'{sat_name}_#Omited for privacy reasons.npy'), data_to_save)
#%%
# Will transform and save the graphical data as gramian images 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from pyts.image import GramianAngularField

data_dir = r"D:\Dataset\ProcessedDataNumpy_NotNormalized"
save_dir = r"D:\Dataset\GramianImages_normalized_0to1"
os.makedirs(save_dir, exist_ok=True)
all_norm_data, all_labels, all_dates = [], [], []

for file_name in os.listdir(data_dir):
    if file_name.endswith('.npy'):
        data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
        norm_data = data['Norm']
        satellite_name = file_name.split('_')[0]
        dates_collected = data.get('Trimmed_UTC', ['unknown_date'] * len(norm_data))  #'Trimmed_UTC' is a key in the data dictionary
        for i, (measurement, date_collected) in enumerate(zip(norm_data, dates_collected)):
            all_norm_data.append(measurement)
            all_labels.append(satellite_name)
            all_dates.append(date_collected)

all_norm_data, all_labels, all_dates = np.array(all_norm_data), np.array(all_labels), np.array(all_dates)

# Encode labels and scale data
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
# Globally normalize data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
all_norm_data_scaled = scaler.fit_transform(all_norm_data)

# Transform data into Gramian Angular Fields and save images
gaf = GramianAngularField(image_size=18)
for i, (measurement, date_collected, satellite_name) in enumerate(zip(all_norm_data_scaled, all_dates, all_labels)):
    gaf_image = gaf.fit_transform(measurement.reshape(1, -1))
    # Ensures the date_collected is a string and replace invalid characters, this was an issue for a minute
    date_collected_str = str(date_collected).replace(':', '-').replace(' ', '_')
    save_path = os.path.join(save_dir, f"{satellite_name}_{date_collected_str}_gaf_image_{i}.npy")
    np.save(save_path, gaf_image)
    print(f"Saved GAF image: {save_path}")  #Debugging 

saved_images = [f for f in os.listdir(save_dir) if f.endswith('.npy')]
print(f"Number of saved GAF images: {len(saved_images)}")
#%%
# This code displays the original data, normalized data, and GAF images for a random sample. This was part of my data preparation
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from pyts.image import GramianAngularField


data_dir = r"D:\Dataset\ProcessedDataNumpy_NotNormalized"
#save_dir = r"D:\Dataset\GramianImages_normalized_0to1"
#os.makedirs(save_dir, exist_ok=True)

all_norm_data, all_labels, all_dates = [], [], []

for file_name in os.listdir(data_dir):
    if file_name.endswith('.npy'):
        data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
        norm_data = data['Norm']
        satellite_name = file_name.split('_')[0]
        dates_collected = data.get('Trimmed_UTC', ['unknown_date'] * len(norm_data)) 
        for i, (measurement, date_collected) in enumerate(zip(norm_data, dates_collected)):
            all_norm_data.append(measurement)
            all_labels.append(satellite_name)
            all_dates.append(date_collected)

all_norm_data, all_labels, all_dates = np.array(all_norm_data), np.array(all_labels), np.array(all_dates)

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
print("Shape of a single original file:", all_norm_data[1].shape)
scaler = MinMaxScaler(feature_range=(0, 1))
all_norm_data_scaled = scaler.fit_transform(all_norm_data)
print("Shape of a single normalized file:", all_norm_data_scaled[1].shape)
# Normalize each data sample individually to be between 0 and 1 per sample and reshape
#scaler = MinMaxScaler(feature_range=(0, 1))
#all_norm_data_scaled = np.array([scaler.fit_transform(sample.reshape(-1, 1)).flatten() for sample in all_norm_data])

def plot_data(original_measurement, normalized_measurement, gaf_image, wavelengths, date_collected, satellite_name):
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))

    axs[0].plot(wavelengths, all_norm_data[0])
    axs[0].set_title(f"Original Non-Normalized Time Series\n{satellite_name} - {date_collected}")
    axs[0].set_xlabel("Wavelengths")
    axs[0].set_ylabel("Norm")

    axs[1].plot(wavelengths, all_norm_data_scaled[0])
    axs[1].set_title(f"Normalized Time Series\n{satellite_name} - {date_collected}")
    axs[1].set_xlabel("Wavelengths")
    axs[1].set_ylabel("Norm")

    r, theta = cartesian_to_polar(wavelengths, normalized_measurement)
    axs[2] = plt.subplot(1, 4, 3, projection='polar')
    axs[2].scatter(theta, r)
    axs[2].set_title(f"Polar Encoded Data\n{satellite_name} - {date_collected}")

    cax = axs[2].imshow(gaf_image[0], cmap='rainbow', origin='lower')
    axs[2].set_title(f"GAF Image\n{satellite_name} - {date_collected}")
    fig.colorbar(cax, ax=axs[2], orientation='vertical', label='Intensity')
    
    plt.tight_layout()
    plt.show()

def cartesian_to_polar(x, y):
    r = np.sqrt(y**2)
    theta = np.arccos(y)
    return r, theta

gaf = GramianAngularField(image_size=128)
for i, (measurement, date_collected, satellite_name) in enumerate(zip(all_norm_data_scaled, all_dates, all_labels)):
    gaf_image = gaf.fit_transform(measurement.reshape(1, -1))
    date_collected_str = str(date_collected).replace(':', '-').replace(' ', '_')
    #save_path = os.path.join(save_dir, f"{satellite_name}_{date_collected_str}_gaf_image_{i}.npy")
    #np.save(save_path, gaf_image)
    #print(f"Saved GAF image: {save_path}")  # Debugging statement
    
    # Plot the original, normalized, and GAF image for a random sample
    if i == 0:  
        wavelengths = np.arange(len(measurement))  #wavelengths are the same for all samples
        plot_data(all_norm_data[i], measurement, gaf_image, wavelengths, date_collected_str, satellite_name)

#%%
# Code to trin the first cnn model on TS data that included augmentations and smote
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = r"D:\Dataset\ProcessedDataNumpy"

all_norm_data = []
all_labels = []

for file_name in os.listdir(data_dir):
    if file_name.endswith('.npy'):
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path, allow_pickle=True).item()
        
        norm_data = data['Norm']
        satellite_name = file_name.split('_')[0]  
        
        for measurement in norm_data:
            all_norm_data.append(measurement)
            all_labels.append(satellite_name)

all_norm_data = np.array(all_norm_data)
all_labels = np.array(all_labels)

if np.any(np.isnan(all_norm_data)) or np.any(np.isinf(all_norm_data)):
    raise ValueError("Data contains NaN or infinite values.")

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

scaler = StandardScaler()
all_norm_data_scaled = scaler.fit_transform(all_norm_data)

pca = PCA(n_components=18)  # 18 components is the best for this dataset from what I found so far.  eigenvalues concur this choice
all_norm_data_pca = pca.fit_transform(all_norm_data_scaled)

# One-hot encode the labels since they start as integers
num_classes = len(np.unique(all_labels_encoded))
all_labels_encoded = to_categorical(all_labels_encoded, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(all_norm_data_pca, all_labels_encoded, test_size=0.2, stratify=all_labels_encoded, random_state=42)

# These model params along with the filters were found using Bayesian optimization
learning_rate = 0.000537758575336864 
dropout_rate = 0.42
l2_reg = 0.004352826238624662 
batch_size = 40 

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, val_index in kfold.split(X_train, np.argmax(y_train, axis=1)):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, np.argmax(y_train_fold, axis=1))
    y_train_resampled = to_categorical(y_train_resampled, num_classes=num_classes)

    noise_factor = 0.02
    scale_factor = np.random.uniform(0.8, 1.1, X_train_resampled.shape)
    X_train_resampled_augmented = X_train_resampled * scale_factor + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_resampled.shape)

    # Reshaping data for conv1D layers
    X_train_resampled_augmented = X_train_resampled_augmented.reshape((X_train_resampled_augmented.shape[0], X_train_resampled_augmented.shape[1], 1))
    X_val_fold = X_val_fold.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1))

    # Define the model with batch normalization
    model = Sequential()
    model.add(Conv1D(filters=53, kernel_size=3, activation='relu', input_shape=(X_train_resampled_augmented.shape[1], 1), kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=104, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=176, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(104, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(53, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))


    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-6)

    history = model.fit(X_train_resampled_augmented, y_train_resampled, epochs=150, batch_size=batch_size, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping, reduce_lr], verbose=0)

    max_val_accuracy = max(history.history['val_accuracy'])
    print(f"Highest validation accuracy for this fold: {max_val_accuracy:.4f}")

    val_predictions = model.predict(X_val_fold)
    val_predictions = np.argmax(val_predictions, axis=1)
    y_val_labels = np.argmax(y_val_fold, axis=1)
    accuracy = accuracy_score(y_val_labels, val_predictions)
    accuracies.append(accuracy)

print(f"Mean accuracy: {np.mean(accuracies)}")
print(f"Standard deviation of accuracy: {np.std(accuracies)}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

test_predictions = model.predict(X_test)
test_predictions = np.argmax(test_predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_test_labels, test_predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test_labels, test_predictions, target_names=label_encoder.classes_))

#%%
# Code to train the second CNN model on GAF images
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
# Started using the gpu and added the line below
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_label(label):
    return label.strip("[]'").lower()

original_dir = r"/home/jtk1704/Downloads/GramianImages_normalized_per_sample_0to1/"
saved_images = [f for f in os.listdir(original_dir) if f.endswith('.npy')]

X_gaf = []
all_labels = []
for file_name in saved_images:
    gaf_image = np.load(os.path.join(original_dir, file_name))
    X_gaf.append(gaf_image)
    label = clean_label(file_name.split('_')[0])
    all_labels.append(label)

X_gaf = np.array(X_gaf)
X_gaf = X_gaf.reshape(-1, 1, 18, 18)  

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

assert len(X_gaf) == len(all_labels_encoded), "Mismatch in number of samples between X_gaf and all_labels_encoded"

X_train, X_test, y_train, y_test = train_test_split(X_gaf, all_labels_encoded, test_size=0.2, stratify=all_labels_encoded, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

n_splits = 5

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []
all_fold_preds = []
all_fold_labels = []

best_val_accuracy = 0.0
best_model_path = r'/home/jtk1704/Downloads/FALL24Research_and_ML/best_cnn_model_GAF_TEST_FINAL.pth'
final_model_path = r'/home/jtk1704/Downloads/FALL24Research_and_ML/final_cnn_model_GAF_TEST_FINAL.pth'

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    print(f"Fold {fold + 1}/{kf.n_splits}")

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    train_dataset = AugmentedDataset(X_train_fold, y_train_fold)
    val_dataset = AugmentedDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    num_classes = len(np.unique(all_labels_encoded))
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-4) # Regularization added
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)  # Learning rate scheduler
    
    # Training loop
    num_epochs = # Enter value here
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_train_samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_train_samples += inputs.size(0)  
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Number of training samples: {num_train_samples}")
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Decided to check the models with the "best" weights and "final" weights
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(val_loss)

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

    fold_preds = []
    fold_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            fold_preds.extend(predicted.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())
    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_labels)

torch.save(model.state_dict(), final_model_path)
print(f"Final model saved with validation accuracy: {val_accuracy:.2f}%")

cm = confusion_matrix(all_fold_labels, all_fold_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis')
plt.xticks(rotation=90)  # Rotate x-axis labels for clarity
plt.title('Confusion Matrix (Cross-Validation)')
plt.show()

print(classification_report(all_fold_labels, all_fold_preds, target_names=label_encoder.classes_))

mean_train_accuracy = np.mean([np.mean(acc) for acc in all_train_accuracies])
std_train_accuracy = np.std([np.mean(acc) for acc in all_train_accuracies])
mean_val_accuracy = np.mean([np.mean(acc) for acc in all_val_accuracies])
std_val_accuracy = np.std([np.mean(acc) for acc in all_val_accuracies])

print(f"Mean Training Accuracy: {mean_train_accuracy:.2f}% +/- {std_train_accuracy:.2f}%")
print(f"Mean Validation Accuracy: {mean_val_accuracy:.2f}% +/- {std_val_accuracy:.2f}%")

plt.figure(figsize=(12, 5))
for fold in range(n_splits):
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), all_train_accuracies[fold], label=f'Train Accuracy Fold {fold + 1}')
    plt.plot(range(1, num_epochs + 1), all_val_accuracies[fold], label=f'Validation Accuracy Fold {fold + 1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Epoch')
plt.legend()

for fold in range(n_splits):
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), all_train_losses[fold], label=f'Train Loss Fold {fold + 1}')
    plt.plot(range(1, num_epochs + 1), all_val_losses[fold], label=f'Validation Loss Fold {fold + 1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()

test_dataset = AugmentedDataset(X_test, y_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.load_state_dict(torch.load(final_model_path))
model.eval()
test_preds_final = []
test_labels_final = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds_final.extend(predicted.cpu().numpy())
        test_labels_final.extend(labels.cpu().numpy())

cm_final = confusion_matrix(test_labels_final, test_preds_final)
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=label_encoder.classes_)
disp_final.plot(cmap='viridis')
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.title('Confusion Matrix (Testing Set) - Final Weights')
plt.show()

print("Classification Report (Testing Set) - Final Weights")
print(classification_report(test_labels_final, test_preds_final, target_names=label_encoder.classes_))

model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds_best = []
test_labels_best = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds_best.extend(predicted.cpu().numpy())
        test_labels_best.extend(labels.cpu().numpy())

cm_best = confusion_matrix(test_labels_best, test_preds_best)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=label_encoder.classes_)
disp_best.plot(cmap='viridis')
plt.xticks(rotation=90)  
plt.title('Confusion Matrix (Testing Set) - Best Weights')
plt.show()

print("Classification Report (Testing Set) - Best Weights")
print(classification_report(test_labels_best, test_preds_best, target_names=label_encoder.classes_))
#%%
# This code is to train the third CNN model on the GAF images that were generated by the CVAE
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# This augmentation is not used in this model, it was tested with on some iterations but not used n the final model
class AugmentedDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment:
            noise_factor = 0.02  
            noise = noise_factor * np.random.randn(*x.shape)
            x = x + noise

            shift = np.random.randint(-1, 2)  
            x = np.roll(x, shift, axis=-1)

        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.long).to(device)

def clean_label(label):
    return label.strip("[]'").lower()

original_dir = r"/home/jtk1704/Downloads/GramianImages_normalized_0to1/"
saved_images = [f for f in os.listdir(original_dir) if f.endswith('.npy')]

X_gaf = []
all_labels = [] 
for file_name in saved_images:
    gaf_image = np.load(os.path.join(original_dir, file_name))
    X_gaf.append(gaf_image)
    label = clean_label(file_name.split('_')[0])
    all_labels.append(label)

X_gaf = np.array(X_gaf)
X_gaf = X_gaf.reshape(-1, 1, 18, 18) 

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

assert len(X_gaf) == len(all_labels_encoded), "Mismatch in number of samples between X_gaf and all_labels_encoded"

generated_dir = r"/home/jtk1704/Downloads/GAF_CVAE_VGG16_FINALsynthetic_images_ACTUAL/"

print(f"Generated directory: {generated_dir}")

if not os.path.exists(generated_dir):
    print(f"Directory does not exist: {generated_dir}")
else:
    print(f"Directory exists: {generated_dir}")

all_files = os.listdir(generated_dir)
print(f"All files in generated directory: {all_files}")

X_generated = []
generated_labels = []  

generated_images = [f for f in all_files if f.endswith('.npy')]

for file_name in generated_images:
    file_path = os.path.join(generated_dir, file_name)
    #print(f"Loading file: {file_path}")  # Debugging: Print the file path
    gaf_image = np.load(file_path)
    X_generated.append(gaf_image)
    # Extract the label from the file name (assuming the label is part of the file name)
    label = clean_label(file_name.split('_')[2])  # Adjusted index to match the label position
    generated_labels.append(label)

#print(f"Unique labels in generated data: {np.unique(generated_labels)}")

X_generated = np.array(X_generated)
X_generated = X_generated.reshape(-1, 1, 18, 18)

# These were encded the same as the original data labels
generated_labels_encoded = label_encoder.transform(generated_labels)

X_train_orig, X_test, y_train_orig, y_test = train_test_split(X_gaf, all_labels_encoded, test_size=0.5, stratify=all_labels_encoded, random_state=42)

X_train_combined = np.concatenate((X_train_orig, X_generated), axis=0)
y_train_combined = np.concatenate((y_train_orig, generated_labels_encoded), axis=0)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

n_splits = 10

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []
all_fold_preds = []
all_fold_labels = []

best_val_accuracy = 0.0 # There were some isses with the val accuracies and i tinkered with this value when it would crash to 0
best_model_path = r'/home/jtk1704/Downloads/FALL24Research_and_ML/best_cnn_model_GAF_VGG16_50perorig_FINAL.pth'
final_model_path = r'/home/jtk1704/Downloads/FALL24Research_and_ML/final_cnn_model_GAF_VGG16_50perorig_FINAL.pth'

for fold, (train_index, val_index) in enumerate(kf.split(X_train_combined, y_train_combined)):
    print(f"Fold {fold + 1}/{kf.n_splits}")

    X_train_fold, X_val_fold = X_train_combined[train_index], X_train_combined[val_index]
    y_train_fold, y_val_fold = y_train_combined[train_index], y_train_combined[val_index]

    train_dataset = AugmentedDataset(X_train_fold, y_train_fold, augment=False)
    val_dataset = AugmentedDataset(X_val_fold, y_val_fold, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_classes = len(np.unique(all_labels_encoded))
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True) 
    
    num_epochs = 150
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_train_samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_train_samples += inputs.size(0)
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Number of training samples: {num_train_samples}")
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device) 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_loss)

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

    fold_preds = []
    fold_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            fold_preds.extend(predicted.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())
    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_labels)

torch.save(model.state_dict(), final_model_path)
print(f"Final model saved with validation accuracy: {val_accuracy:.2f}%")

cm = confusion_matrix(all_fold_labels, all_fold_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis')
plt.xticks(rotation=90)  
plt.title('Confusion Matrix (Cross-Validation)')
plt.show()

print(classification_report(all_fold_labels, all_fold_preds, target_names=label_encoder.classes_))

mean_train_accuracy = np.mean([np.mean(acc) for acc in all_train_accuracies])
std_train_accuracy = np.std([np.mean(acc) for acc in all_train_accuracies])
mean_val_accuracy = np.mean([np.mean(acc) for acc in all_val_accuracies])
std_val_accuracy = np.std([np.mean(acc) for acc in all_val_accuracies])

print(f"Mean Training Accuracy: {mean_train_accuracy:.2f}% +/- {std_train_accuracy:.2f}%")
print(f"Mean Validation Accuracy: {mean_val_accuracy:.2f}% +/- {std_val_accuracy:.2f}%")

plt.figure(figsize=(12, 5))
for fold in range(n_splits):
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), all_train_accuracies[fold], label=f'Train Accuracy Fold {fold + 1}')
    plt.plot(range(1, num_epochs + 1), all_val_accuracies[fold], label=f'Validation Accuracy Fold {fold + 1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Epoch')
plt.legend()

for fold in range(n_splits):
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), all_train_losses[fold], label=f'Train Loss Fold {fold + 1}')
    plt.plot(range(1, num_epochs + 1), all_val_losses[fold], label=f'Validation Loss Fold {fold + 1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch')
plt.legend()

plt.tight_layout()
#plt.savefig(r'/home/jtk1704/Downloads/FALL24Research_and_ML/CNN_LR_CVAEImages/loss_accuracy_graphs.png')
plt.show()

test_dataset = AugmentedDataset(X_test, y_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.load_state_dict(torch.load(final_model_path))
model.eval()
test_preds_final = []
test_labels_final = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds_final.extend(predicted.cpu().numpy())
        test_labels_final.extend(labels.cpu().numpy())

cm_final = confusion_matrix(test_labels_final, test_preds_final)
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=label_encoder.classes_)
disp_final.plot(cmap='viridis')
plt.xticks(rotation=90) 
plt.title('Confusion Matrix (Testing Set) - Final Weights')
#plt.savefig(r'/home/jtk1704/Downloads/FALL24Research_and_ML/CNN_LR_CVAEImages/confusion_matrix_testfinal.png')
plt.show()

print("Classification Report (Testing Set) - Final Weights")
print(classification_report(test_labels_final, test_preds_final, target_names=label_encoder.classes_))

model.load_state_dict(torch.load(best_model_path))
model.eval()
test_preds_best = []
test_labels_best = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds_best.extend(predicted.cpu().numpy())
        test_labels_best.extend(labels.cpu().numpy())

cm_best = confusion_matrix(test_labels_best, test_preds_best)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=label_encoder.classes_)
disp_best.plot(cmap='viridis')
plt.xticks(rotation=90) 
plt.title('Confusion Matrix (Testing Set) - Best Weights')
#plt.savefig(r'/home/jtk1704/Downloads/FALL24Research_and_ML/CNN_LR_CVAEImages/confusion_matrix_testbest.png')
plt.show()

print("Classification Report (Testing Set) - Best Weights")
print(classification_report(test_labels_best, test_preds_best, target_names=label_encoder.classes_))
#%%
# Code to train the TS CNN model with the data generated by the CVAE
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

save_dir = r"/home/jtk1704/Downloads/ProcessedDataNumpy_notfiltered_norm"
saved_files = [f for f in os.listdir(save_dir) if f.endswith('.npy')]

X_ts = []
all_labels = []
for file_name in saved_files:
    data_array = np.load(os.path.join(save_dir, file_name), allow_pickle=True)
    if data_array.ndim == 2 and data_array.size > 0:
        for ts_data in data_array:
            ts_data = np.array(ts_data)
            if ts_data.ndim == 1 and ts_data.size > 0:
                ts_data = ts_data.reshape(1, -1)
                X_ts.append(ts_data)
                label = file_name.split('_')[0].strip("[]'").lower()
                all_labels.append(label)

max_length = max(ts.shape[1] for ts in X_ts)
X_ts_padded = [np.pad(ts, ((0, 0), (0, max_length - ts.shape[1])), 'constant') for ts in X_ts]

X_ts = np.array(X_ts_padded)
X_ts = X_ts.reshape(-1, 1, X_ts.shape[2])

generated_dir = r"/home/jtk1704/Downloads/TS_synthetic_data_w_skip_augmenteddata_incode_actual_l350n01m05_ACTUAL/"
X_generated = []
generated_labels = []

for file_name in os.listdir(generated_dir):
    if file_name.endswith('.npy'):
        label = file_name.split('_')[0].strip("[]'").lower()
        file_path = os.path.join(generated_dir, file_name)
        data = np.load(file_path)
        X_generated.append(data)
        generated_labels.append(label)

X_generated = np.array(X_generated)
X_generated = X_generated.reshape(-1, 1, X_generated.shape[2])

print("Unique labels in original data:", np.unique(all_labels))
print("Unique labels in generated data:", np.unique(generated_labels))

X_combined = np.concatenate((X_ts, X_generated), axis=0)
all_labels_combined = all_labels + generated_labels

label_encoder = LabelEncoder()
all_labels_encoded_combined = label_encoder.fit_transform(all_labels_combined)

num_classes = len(np.unique(all_labels_encoded_combined))
print(f"Number of unique labels: {num_classes}")

all_labels_encoded = all_labels_encoded_combined[:len(all_labels)]
generated_labels_encoded = all_labels_encoded_combined[len(all_labels):]

with open(r'/home/jtk1704/Downloads/label_encoder_timeseriesCVAE.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

X_train_orig, X_test, y_train_orig, y_test = train_test_split(X_ts, all_labels_encoded, test_size=0.2, stratify=all_labels_encoded, random_state=42)

X_train_combined = np.concatenate((X_train_orig, X_generated), axis=0)
y_train_combined = np.concatenate((y_train_orig, generated_labels_encoded), axis=0)

num_samples, num_channels, num_features = X_train_combined.shape
X_train_combined_flattened = X_train_combined.reshape(num_samples, -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Fit the scaler on the combined training data Scalar must be done before PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined_flattened)

X_test_scaled = scaler.transform(X_test_flattened)

pca = PCA(n_components=18)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_train_pca = X_train_pca.reshape(X_train_pca.shape[0], 1, X_train_pca.shape[1])
X_test_pca = X_test_pca.reshape(X_test_pca.shape[0], 1, X_test_pca.shape[1])

y_train_encoded = np.eye(num_classes)[y_train_combined]
y_test_encoded = np.eye(num_classes)[y_test]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = TimeSeriesDataset(X_train_pca, y_train_encoded)

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 53, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(53)
        self.conv2 = nn.Conv1d(53, 104, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(104)
        self.conv3 = nn.Conv1d(104, 176, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(176)
        self.fc1 = nn.Linear(176 * 18, 104) # Multiply by the pca amount (18 here)
        self.bn4 = nn.BatchNorm1d(104)
        self.fc2 = nn.Linear(104, 53)
        self.bn5 = nn.BatchNorm1d(53)
        self.fc3 = nn.Linear(53, num_classes)
        self.dropout = nn.Dropout(0.42)
        self.leaky_relu = nn.ReLU()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

criterion = nn.CrossEntropyLoss()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_path = '/home/jtk1704/Downloads/TS_CNN_PYTORCH_PCA18_TEST_best_model_weights.pth'
final_model_path = '/home/jtk1704/Downloads/TS_CNN_PYTORCH_PCA18_TEST_final_model_weights.pth'

all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []
all_fold_preds = []
all_fold_labels = []
fold_accuracies = []
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_pca, np.argmax(y_train_encoded, axis=1))):
    print(f'Fold {fold + 1}')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=40, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=40, shuffle=False)
    
    model = CNNModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.000537758575336864)
    num_epochs = 100
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, torch.argmax(labels, dim=1))

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Train Acc: {100 * correct_train / total_train:.2f}%, Val Acc: {100 * correct_val / total_val:.2f}%')

        if val_accuracies[-1] > best_val_accuracy:
            best_val_accuracy = val_accuracies[-1]
            torch.save(model.state_dict(), best_model_path)

    fold_accuracies.append(best_val_accuracy)
    fold_histories.append((train_losses, val_losses, train_accuracies, val_accuracies))

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

    fold_preds = []
    fold_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            fold_preds.extend(predicted.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())
    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_labels)

mean_train_accuracy = np.mean([np.mean(acc) for acc in all_train_accuracies])
std_train_accuracy = np.std([np.mean(acc) for acc in all_train_accuracies])
mean_val_accuracy = np.mean([np.mean(acc) for acc in all_val_accuracies])
std_val_accuracy = np.std([np.mean(acc) for acc in all_val_accuracies])

print(f"Mean Training Accuracy: {mean_train_accuracy:.2f}% +/- {std_train_accuracy:.2f}%")
print(f"Mean Validation Accuracy: {mean_val_accuracy:.2f}% +/- {std_val_accuracy:.2f}%")
torch.save(model.state_dict(), final_model_path)

plt.figure(figsize=(12, 8))
for fold, (train_losses, val_losses, train_accuracies, val_accuracies) in enumerate(fold_histories):
    plt.subplot(5, 2, 2*fold+1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(5, 2, 2*fold+2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()

test_dataset = TimeSeriesDataset(X_test_pca, y_test_encoded)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

model.load_state_dict(torch.load(final_model_path))
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, torch.argmax(labels, dim=1))

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == torch.argmax(labels, dim=1)).sum().item()
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())

print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct_test / total_test:.2f}%')

conf_matrix_test_final = confusion_matrix(all_test_labels, all_test_preds)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_test_final, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Data (Final Weights)')
plt.show()

print("Classification Report - Test Data (Final Weights)")
print(classification_report(all_test_labels, all_test_preds, target_names=label_encoder.classes_[:num_classes]))

model.load_state_dict(torch.load(best_model_path))
model.eval()
correct_test_best = 0
all_test_preds_best = []

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        correct_test_best += (predicted == torch.argmax(labels, dim=1)).sum().item()
        all_test_preds_best.extend(predicted.cpu().numpy())

print(f'Test Accuracy (Best Weights): {100 * correct_test_best / total_test:.2f}%')

conf_matrix_test_best = confusion_matrix(all_test_labels, all_test_preds_best)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_test_best, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Data (Best Weights)')
plt.show()

print("Classification Report - Test Data (Best Weights)")
print(classification_report(all_test_labels, all_test_preds_best, target_names=label_encoder.classes_[:num_classes]))
#%%
# This code is the CVAE for TS data with skip connections
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from imblearn.over_sampling import SMOTE
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import json
import random
# Above i loaded in multiple libraries because i was checking various metrics during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = r"/home/jtk1704/Downloads/ProcessedDataNumpy_notfiltered_norm"
saved_files = [f for f in os.listdir(save_dir) if f.endswith('.npy')]

X_ts = []
all_labels = []  
for file_name in saved_files:
    data_array = np.load(os.path.join(save_dir, file_name), allow_pickle=True)
    if data_array.ndim == 2 and data_array.size > 0: 
        for ts_data in data_array:
            ts_data = np.array(ts_data)  
            if ts_data.ndim == 1 and ts_data.size > 0:  
                ts_data = ts_data.reshape(1, -1)  #Reshape to (1, 501)
                X_ts.append(ts_data)
                label = file_name.split('_')[0].strip("[]").lower() 
                all_labels.append(label)

# This ensures all time series data have the same shape by padding sequences
max_length = max(ts.shape[1] for ts in X_ts)
X_ts_padded = [np.pad(ts, ((0, 0), (0, max_length - ts.shape[1])), 'constant') for ts in X_ts]

X_ts = np.array(X_ts_padded)
X_ts = X_ts.reshape(-1, 1, X_ts.shape[2])

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

with open(r'/home/jtk1704/Downloads/label_encoder_timeseriesCVAE.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

X_temp, X_test, y_temp, y_test = train_test_split(X_ts, all_labels_encoded, test_size=0.5, stratify=all_labels_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=42)  
unique_test_indices = np.unique(np.arange(len(X_test)))
X_test = X_test[unique_test_indices]
y_test = y_test[unique_test_indices]

print(f"Test dataset size: {len(X_test)}, Labels size: {len(y_test)}")

class CustomDataset(Dataset):
    def __init__(self, data, labels, augment=False, num_augmentations=60):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.Lambda(lambda ts: ts + torch.randn_like(ts) * 0.02),  # noise
                transforms.Lambda(lambda ts: ts * (0.8 + 0.4 * torch.rand_like(ts))),  # scaling
                transforms.Lambda(lambda ts: ts * (torch.rand_like(ts) > 0.05).float()),  # sensor dropout 
            ], p=0.5)
        ])

    def __len__(self):
        if self.augment:
            return len(self.data) * self.num_augmentations
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.augment:
            original_idx = idx // self.num_augmentations
            x = self.data[original_idx]
            y = self.labels[original_idx]
        else:
            x = self.data[idx]
            y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)  # Convert to PyTorch tensor

        if self.augment:
            x = self.transform(x)

        return x, torch.tensor(y, dtype=torch.long)

# Had these up here to fiddle with them and try to optimize my model
num_classes = len(np.unique(all_labels_encoded))
latent_dim = 350  
noise_level = 1  
kld_weight = 1  
learning_rate = 1e-4  

class CVAE(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, latent_dim=latent_dim, num_classes=num_classes):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder block
        self.conv1 = nn.Conv1d(input_channels + num_classes, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc_mu = nn.Linear(512 * 63, latent_dim)  
        self.fc_logvar = nn.Linear(512 * 63, latent_dim)  

        # Decoder block
        self.fc3 = nn.Linear(latent_dim + num_classes, 512 * 63)  
        self.deconv1 = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.deconv2 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm1d(128)
        self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose1d(64, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1)  

        self.dropout = nn.Dropout(0.6)
        self.leakyrelu = nn.PReLU()  # These all say leaky relu but i changed them to PReLU and didnt bother with adjusting the name, i apologize for the confusion

    def encode(self, x, c):
        if x.ndim == 2:  # There were many size mismatch issues and a few of these lines were to address that
            x = x.unsqueeze(1)
        c = c.view(-1, self.num_classes, 1).expand(-1, -1, x.size(2))
        x = torch.cat([x, c], dim=1)
        h1 = self.leakyrelu(self.bn1(self.conv1(x)))
        h2 = self.leakyrelu(self.bn2(self.conv2(h1)))
        h3 = self.leakyrelu(self.bn3(self.conv3(h2)))
        h3 = h3.view(h3.size(0), -1)
        h3 = self.dropout(h3)
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        logvar = torch.clamp(logvar, min=-10, max=10) # Log values were causing issues so i clamped them
        return mu, logvar, h1, h2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_level  
        return mu + eps * std

    def decode(self, z, c, h1=None, h2=None):
        z = torch.cat([z, c], dim=1)
        h3 = self.leakyrelu(self.fc3(z))
        h3 = h3.view(h3.size(0), 512, 63) 
        h4 = self.leakyrelu(self.bn5(self.deconv1(h3)))
        if h2 is not None:
            h4 = h4[:, :, :h2.size(2)]
            h4 = h4 + h2  
        h5 = self.leakyrelu(self.bn6(self.deconv2(h4)))
        if h1 is not None:
            h5 = h5[:, :, :h1.size(2)]
            h5 = h5 + h1  
        h6 = self.leakyrelu(self.bn7(self.deconv3(h5)))
        h7 = torch.sigmoid(self.deconv4(h6))  # Used sigmoid to ensure output is in range [0, 1]
        return h7

    def forward(self, x, c):
        if x.ndim == 2:  
            x = x.unsqueeze(1)
        mu, logvar, h1, h2 = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, c, h1, h2)
        return decoded, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    RE = nn.functional.mse_loss(recon_x, x, reduction='sum')  # Used MSE as explained in paper it is almost identical to maximizing the log loss (up to a constant) when assuming gaussian distributions
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RE + kld_weight * KLD, RE, KLD  

cvae = CVAE(num_classes=num_classes).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)

train_dataset = CustomDataset(X_train, y_train, augment=True, num_augmentations=60) # Adds 60 augmented samples per orignal sample to this
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print(f"Number of data points in training dataset: {len(train_dataset)}")

val_dataset = CustomDataset(X_val, y_val, augment=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

epochs = 15  # This didnt need a large number to converge
train_loss = []
val_loss = []
train_kld = []
val_kld = []
val_mse = []
val_wasserstein = []

for epoch in range(epochs):
    cvae.train()
    train_loss_epoch = 0
    train_kld_epoch = 0
    train_recon_epoch = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
        recon_batch, mu, logvar = cvae(data, labels_one_hot)
        recon_batch = recon_batch[:, :, :data.shape[2]] 
        loss, RE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0) # Gradient clipping used to prevent exploding gradients and eep training stable
        
        train_loss_epoch += loss.item()
        train_kld_epoch += KLD.item()
        train_recon_epoch += RE.item()
        optimizer.step()

    train_loss.append(train_loss_epoch / len(train_loader.dataset))
    train_kld.append(train_kld_epoch / len(train_loader.dataset))
    train_recon = train_recon_epoch / len(train_loader.dataset)

    cvae.eval()
    val_loss_epoch = 0
    val_kld_epoch = 0
    val_recon_epoch = 0
    mse_epoch = 0
    wasserstein_epoch = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
            recon_batch, mu, logvar = cvae(data, labels_one_hot)
            recon_batch = recon_batch[:, :, :data.shape[2]]  
            loss, RE, KLD = loss_function(recon_batch, data, mu, logvar)
            val_loss_epoch += loss.item()
            val_kld_epoch += KLD.item()
            val_recon_epoch += RE.item()
            mse_batch = ((recon_batch - data) ** 2).mean(dim=[1, 2])
            wasserstein_batch = [wasserstein_distance(recon_batch[i].cpu().numpy().squeeze(), data[i].cpu().numpy().squeeze()) for i in range(data.size(0))]
            mse_epoch += mse_batch.sum().item()
            wasserstein_epoch += sum(wasserstein_batch)

    val_loss.append(val_loss_epoch / len(val_loader.dataset))
    val_kld.append(val_kld_epoch / len(val_loader.dataset))
    val_recon = val_recon_epoch / len(val_loader.dataset)
    val_mse.append(mse_epoch / len(val_loader.dataset))
    val_wasserstein.append(wasserstein_epoch / len(val_loader.dataset))

    print(f"Epoch {epoch + 1}:")
    print(f"  Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
    print(f"  Train KLD: {train_kld[-1]:.4f}, Val KLD: {val_kld[-1]:.4f}")
    print(f"  Train Recon: {train_recon:.4f}, Val Recon: {val_recon:.4f}")
    print(f"  Val MSE: {val_mse[-1]:.4f}, Val Wasserstein: {val_wasserstein[-1]:.4f}")

combined_score = val_mse[-1] + val_wasserstein[-1]  # Used as a metric but not discussed in detail in the paper. This is a combination of the two metrics was found to be more beneficial and meaningful that using ssim, or cosine similarity with mse.

print(f"Combined Score: {combined_score}")

save_path = r'/home/jtk1704/Downloads/cvae_model_timeseries_w_skip_augmenteddata_incode_actual_l350n01m05.pth'
torch.save(cvae.state_dict(), save_path)
print(f'Model saved to {save_path}')

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot(train_kld, label='Training KLD')
plt.plot(val_kld, label='Validation KLD')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with KLD')
plt.legend()
plt.show()

test_dataset = CustomDataset(X_test, y_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"Number of data points in test dataset: {len(test_dataset)}")

test_loss = 0
test_re_loss = 0
test_kld_loss = 0
mse_values = []

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        recon_batch, mu, logvar = cvae(data, labels_one_hot)
        recon_batch = recon_batch[:, :, :X_ts.shape[2]]  
        loss, RE, kld = loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.item()
        test_re_loss += RE.item()
        test_kld_loss += kld.item()

        mse_batch = ((recon_batch - data) ** 2).mean(dim=[1, 2])
        mse_values.extend(mse_batch.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_re_loss /= len(test_loader.dataset)
test_kld_loss /= len(test_loader.dataset)
avg_mse = np.mean(mse_values)

print(f'Test Loss: {test_loss:.4f}, Test RE: {test_re_loss:.4f}, Test KLD: {test_kld_loss:.4f}')
print(f'Average MSE: {avg_mse}')

output_dir = r"/home/jtk1704/Downloads/TS_synthetic_data_w_skip_augmenteddata_incode_actual_l350n01m05/"
os.makedirs(output_dir, exist_ok=True)

def generate_synthetic_data(cvae, num_samples_per_class=60):  
    samples = []
    labels = []
    with torch.no_grad():
        for i in range(num_classes):
            labels_one_hot = torch.nn.functional.one_hot(torch.tensor([i] * num_samples_per_class), num_classes=num_classes).float().to(device)
            z = torch.randn(num_samples_per_class, cvae.latent_dim).to(device)
            sample = cvae.decode(z, labels_one_hot).cpu().numpy()
            sample = sample[:, :, :X_ts.shape[2]]  
            sample = sample.astype(np.float64)  # Convert to float64 for higher numerical precision for our CNN model to use
            samples.append(sample)
            labels.extend([i] * num_samples_per_class)
 
    return np.concatenate(samples), np.array(labels)

synthetic_data, labels_resampled = generate_synthetic_data(cvae, num_samples_per_class=60)
for i, (data, label) in enumerate(zip(synthetic_data, labels_resampled)):
    label_name = label_encoder.inverse_transform([label])[0]
    file_name = f"{label_name}_synthetic_{i}.npy"
    file_path = os.path.join(output_dir, file_name)
    np.save(file_path, data)

print(f"Generated {synthetic_data.shape[0]} synthetic samples.")
print(f"Saved synthetic data to {output_dir}")
# This allowed me to choose a random sample from each class to compare to the original data
def select_random_timeseries_per_label(timeseries, labels, num_classes):
    selected_timeseries = []
    selected_labels = []
    for i in range(num_classes):
        indices = [index for index, label in enumerate(labels) if label == i]
        random_index = random.choice(indices)
        selected_timeseries.append(timeseries[random_index])
        selected_labels.append(labels[random_index])
    return np.array(selected_timeseries), np.array(selected_labels)
for data, label in test_loader:
    print(f"Batch data size: {data.size()}, Batch label size: {label.size()}")
    break  # Checking batch information to verify 

def extract_latent_space(cvae, data_loader):
    cvae.eval()
    latent_space = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=num_classes).float().to(device)
            mu, logvar, _, _ = cvae.encode(data, label_one_hot)
            latent_space.append(mu.cpu().numpy())
            labels.append(label.cpu().numpy())
            print(f"Latent space batch shape: {mu.shape}, Labels batch shape: {label.shape}")

    return np.concatenate(latent_space), np.concatenate(labels)

original_latent_space, original_labels = extract_latent_space(cvae, test_loader)
assert original_latent_space.shape[0] == original_labels.shape[0], "Mismatch in test data dimensions!"
print(f"Number of data points in original latent space: {original_latent_space.shape[0]}")

# Above was used to save the synthetic data, this is used to extrract its data for some charts below
synthetic_data, synthetic_labels = generate_synthetic_data(cvae)
synthetic_dataset = CustomDataset(synthetic_data, synthetic_labels, augment=False)
synthetic_loader = DataLoader(synthetic_dataset, batch_size=64, shuffle=False)
synthetic_latent_space, synthetic_labels = extract_latent_space(cvae, synthetic_loader)

def plot_latent_space_histograms(original_latent_space, original_labels, synthetic_latent_space, synthetic_labels, num_classes):
    plt.figure(figsize=(20, 10))
    for i in range(num_classes):
        plt.subplot(2, num_classes, i + 1)
        plt.hist(original_latent_space[original_labels == i].flatten(), bins=30, alpha=0.5, label='Original')
        plt.title(f'Original Label {i}')
        plt.subplot(2, num_classes, i + 1 + num_classes)
        plt.hist(synthetic_latent_space[synthetic_labels == i].flatten(), bins=30, alpha=0.5, label='Synthetic')
        plt.title(f'Synthetic Label {i}')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_latent_space_histograms(original_latent_space, original_labels, synthetic_latent_space, synthetic_labels, num_classes)

selected_original_timeseries, selected_original_labels = select_random_timeseries_per_label(X_test, y_test, num_classes)
selected_synthetic_timeseries, selected_synthetic_labels = select_random_timeseries_per_label(synthetic_data, labels_resampled, num_classes)

# To show the generated and original images side by side
plt.figure(figsize=(20, 10))
for i in range(num_classes):

    ax1 = plt.subplot(4, 10, i + 1)
    plt.plot(selected_original_timeseries[i].squeeze())
    ax1.set_title(f'Original {label_encoder.inverse_transform([selected_original_labels[i]])[0]} ({i})', fontsize=8)
    plt.axis('off')
    
    ax2 = plt.subplot(4, 10, i + 1 + num_classes)
    plt.plot(selected_synthetic_timeseries[i].squeeze())
    ax2.set_title(f'Synthetic {label_encoder.inverse_transform([selected_synthetic_labels[i]])[0]} ({i})', fontsize=8)
    plt.axis('off')

#%%
#The following code is the CVAE for GAF data with the VGG16 used to extract features for the loss function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
import pickle
import torchvision.transforms as transforms
import torchvision.models as models
import seaborn as sns
import random
from imblearn.over_sampling import SMOTE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = r"/home/jtk1704/Downloads/GramianImages_normalized_0to1/"
saved_images = [f for f in os.listdir(save_dir) if f.endswith('.npy')]

X_gaf = []
all_labels = []  # Define the all_labels list
for file_name in saved_images:
    gaf_image = np.load(os.path.join(save_dir, file_name))
    X_gaf.append(gaf_image)
    label = file_name.split('_')[0].lower()  # Convert label to lowercase to make it easier to work with
    all_labels.append(label)

X_gaf = np.array(X_gaf)
X_gaf = X_gaf.reshape(-1, 1, 18, 18)  

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

with open(r'/home/jtk1704/Downloads/label_encoder_gramian.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

assert len(X_gaf) == len(all_labels_encoded), "Mismatch in number of samples between X_gaf and all_labels_encoded"

X_temp, X_test, y_temp, y_test = train_test_split(X_gaf, all_labels_encoded, test_size=0.5, stratify=all_labels_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=42)  

print(f"Unique labels in training set: {np.unique(y_train)}")
print(f"Unique labels in validation set: {np.unique(y_val)}")
print(f"Unique labels in test set: {np.unique(y_test)}")

# Shows the index to name mapping for labels
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Showing this here since it is not a usual import
from torchvision.models import VGG16_Weights

# Added loss value for the loss function to use during training
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval() # Using 16 layers based on their pre-trained weights. They are in eval mode to only extract the features
        for param in self.layers.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)  # Repeat grayscale image to 3 channels since that is what VGG16 expects
        y = y.repeat(1, 3, 1, 1)  
        x_features = self.layers(x)
        y_features = self.layers(y)
        return nn.functional.mse_loss(x_features, y_features)

latent_dims = 200
noise_level = 0.232
class CVAE(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, latent_dim=latent_dims, num_classes=20, activation_function=nn.PReLU()):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.actfunct = activation_function

        self.conv1 = nn.Conv2d(input_channels + num_classes, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc_mu = nn.Linear(512 * 3 * 3, latent_dim) 
        self.fc_logvar = nn.Linear(512 * 3 * 3, latent_dim)  

        self.fc3 = nn.Linear(latent_dim + num_classes, 512 * 3 * 3)  
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 

        self.dropout = nn.Dropout(0.5)

    def encode(self, x, c):
        c = c.view(-1, self.num_classes, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        h1 = self.actfunct(self.bn1(self.conv1(x)))
       # print(f"h1 shape: {h1.shape}") # Size mismatches, so print lines added to debug
        h2 = self.actfunct(self.bn2(self.conv2(h1)))
       # print(f"h2 shape: {h2.shape}") 
        h3 = self.actfunct(self.bn3(self.conv3(h2)))
       # print(f"h3 shape: {h3.shape}") 
        h3 = h3.view(h3.size(0), -1)
       # print(f"h3 reshaped: {h3.shape}") 
        h3 = self.dropout(h3)
        return self.fc_mu(h3), self.fc_logvar(h3), h1, h2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_level
        return mu + eps * std

    def decode(self, z, c, h1=None, h2=None):
        z = torch.cat([z, c], dim=1)
        h3 = self.actfunct(self.fc3(z))
       # print(f"h3 shape: {h3.shape}") 
        h3 = h3.view(h3.size(0), 512, 3, 3)  # Adjust input size to match output shape of conv3
       # print(f"h3 reshaped: {h3.shape}") 
        h4 = self.actfunct(self.bn4(self.deconv1(h3)))
       # print(f"h4 shape: {h4.shape}") 
        h5 = self.actfunct(self.bn5(self.deconv2(h4)))
       # print(f"h5 shape: {h5.shape}") 
        h6 = self.actfunct(self.bn6(self.deconv3(h5)))
       # print(f"h6 shape: {h6.shape}") 
        h7 = torch.tanh(self.deconv4(h6))  # Using tanh to ensure output is in range [-1, 1]
       # print(f"h7 shape: {h7.shape}") 
        return h7

    def forward(self, x, c):
        mu, logvar, h1, h2 = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, c, h1, h2)
        return decoded, mu, logvar

def loss_function(recon_x, x, mu, logvar, perceptual_loss):
    RE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    perceptual_loss_value = perceptual_loss(recon_x, x)
    total_loss = RE + 0.179 * KLD + perceptual_loss_value  
    return total_loss, RE, KLD, perceptual_loss_value

import torchvision.transforms as transforms
from imblearn.over_sampling import SMOTE
import torch


class CustomDataset(Dataset):
    def __init__(self, data, labels, augment=False, num_augmentations=60):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.05),  # Add noise
                transforms.Lambda(lambda img: img * (0.8 + 0.4 * torch.rand_like(img))),  # Scaling
                transforms.Lambda(lambda img: self.apply_sensor_dropout(img)),  # Sensor dropout 
            ], p=0.5)
        ])

    def apply_sensor_dropout(self, img):
        min_value = img.min()
        mask = (torch.rand_like(img) > 0.05).float()
        return img * mask + min_value * (1 - mask)#making sure it droupouts to the min value instead of 0

    def __len__(self):
        return len(self.data) * self.num_augmentations if self.augment else len(self.data)

    def __getitem__(self, idx):
        if self.augment:
            original_idx = idx // self.num_augmentations
            x = self.data[original_idx]
            y = self.labels[original_idx]
        else:
            x = self.data[idx]
            y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)  

        if self.augment:
            x = self.transform(x)

        return x, torch.tensor(y, dtype=torch.long)

def decode_features(cvae, features, labels):
    cvae.eval()
    decoded_images = []
    with torch.no_grad():
        for feature, label in zip(features, labels):
            label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).float().unsqueeze(0).to(device)
            feature = torch.tensor(feature).unsqueeze(0).to(device)
            decoded_image = cvae.decode(feature, label_one_hot).cpu().numpy()
            decoded_image = decoded_image[:, :, :18, :18] # Put image back to original size
            decoded_images.append(decoded_image)
    return np.concatenate(decoded_images)

train_dataset = CustomDataset(X_gaf, all_labels_encoded, augment=True, num_augmentations=60)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

num_classes = len(np.unique(all_labels_encoded))
cvae = CVAE(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(cvae.parameters(), lr=0.0008533732689800001)
perceptual_loss = PerceptualLoss().to(device)

epochs = 15
train_loss = []
RE_loss = []
kld_loss = []
perceptual_loss_values = []
val_loss = []
val_RE_loss = []
val_kld_loss = []
val_perceptual_loss_values = []

for epoch in range(epochs):
    cvae.train()
    train_loss_epoch = 0
    RE_loss_epoch = 0
    kld_loss_epoch = 0
    perceptual_loss_epoch = 0
    num_train_samples = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
        data = data.to(device)

        recon_batch, mu, logvar = cvae(data, labels_one_hot)
        recon_batch = recon_batch[:, :, :18, :18] 
        
        loss, RE, kld, perceptual_loss_value = loss_function(recon_batch, data, mu, logvar, perceptual_loss)

        loss.backward()
        train_loss_epoch += loss.item()
        RE_loss_epoch += RE.item()
        kld_loss_epoch += kld.item()
        perceptual_loss_epoch += perceptual_loss_value.item()
        optimizer.step()
        
        num_train_samples += data.size(0)
    
    print(f"Number of training samples in epoch {epoch + 1}: {num_train_samples}")
 
    train_loss.append(train_loss_epoch / len(train_loader.dataset))
    RE_loss.append(RE_loss_epoch / len(train_loader.dataset))
    kld_loss.append(kld_loss_epoch / len(train_loader.dataset))
    perceptual_loss_values.append(perceptual_loss_epoch / len(train_loader.dataset))
    
    val_dataset = CustomDataset(X_val, y_val, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    cvae.eval()
    val_loss_epoch = 0
    val_RE_loss_epoch = 0
    val_kld_loss_epoch = 0
    val_perceptual_loss_epoch = 0
    num_val_samples = 0

    with torch.no_grad():
        for data, labels in val_loader:
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
            data = data.to(device)
            recon_batch, mu, logvar = cvae(data, labels_one_hot)
            recon_batch = recon_batch[:, :, :18, :18] 
        
            loss, RE, kld, perceptual_loss_value = loss_function(recon_batch, data, mu, logvar, perceptual_loss)
            val_loss_epoch += loss.item()
            val_RE_loss_epoch += RE.item()
            val_kld_loss_epoch += kld.item()
            val_perceptual_loss_epoch += perceptual_loss_value.item()
        
            num_val_samples += data.size(0)

    print(f"Number of validation samples: {num_val_samples}")

    val_loss.append(val_loss_epoch / len(val_loader.dataset))
    val_RE_loss.append(val_RE_loss_epoch / len(val_loader.dataset))
    val_kld_loss.append(val_kld_loss_epoch / len(val_loader.dataset))
    val_perceptual_loss_values.append(val_perceptual_loss_epoch / len(val_loader.dataset))
    
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, RE: {RE_loss[-1]:.4f}, KLD: {kld_loss[-1]:.4f}, Perceptual Loss: {perceptual_loss_values[-1]:.4f}') #Create DataLoader for the test data

test_dataset = CustomDataset(X_test, y_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

cvae.eval()
test_loss = 0
test_re_loss = 0
test_kld_loss = 0
test_perceptual_loss = 0
mse_values = []
num_test_samples = 0

with torch.no_grad():
    for data, labels in test_loader:
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)
        data = data.to(device)
        recon_batch, mu, logvar = cvae(data, labels_one_hot)
        recon_batch = recon_batch[:, :, :18, :18] 
        loss, RE, kld, perceptual_loss_value = loss_function(recon_batch, data, mu, logvar, perceptual_loss)
        test_loss += loss.item()
        test_re_loss += RE.item()
        test_kld_loss += kld.item()
        test_perceptual_loss += perceptual_loss_value.item()

        mse_batch = ((recon_batch - data) ** 2).mean(dim=[1, 2, 3])
        mse_values.extend(mse_batch.cpu().numpy())
        
        num_test_samples += data.size(0)

print(f"Number of test samples: {num_test_samples}")

test_loss /= len(test_loader.dataset)
test_re_loss /= len(test_loader.dataset)
test_kld_loss /= len(test_loader.dataset)
test_perceptual_loss /= len(test_loader.dataset)
avg_mse = np.mean(mse_values)

print(f'Test Loss: {test_loss:.4f}, Test RE: {test_re_loss:.4f}, Test KLD: {test_kld_loss:.4f}, Test Perceptual Loss: {test_perceptual_loss:.4f}')
print(f'Average MSE: {avg_mse}')
# Save the model
model_path = r'/home/jtk1704/Downloads/GAF_cvae_VGG16_final_noFineTuning.pth'
torch.save(cvae.state_dict(), model_path)
#The code for the graphs, and plts and synthetic data is the same as the previous code and are omitted here