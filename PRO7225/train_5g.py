"""
PIR - Projet d'Initiation a la recherche @ Telecom Paris
Code 05 - Training the 5G model using the proposed architecture

Author: Alvaro RIBAS
"""

# 0 - Imports ====================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Model
from model_5g import PowerModel

# 1 - Dataset class ======================================================================================================

# Custom Dataset class
class TxPowerDataset(Dataset):
    def __init__(self, building_imgs, antenna_imgs, frequencies, distances, 
                 bs_heights, app_service, target_tx_power):

        self.building_imgs = torch.FloatTensor(building_imgs) if not isinstance(building_imgs, torch.Tensor) else building_imgs
        self.antenna_imgs = torch.FloatTensor(antenna_imgs) if not isinstance(antenna_imgs, torch.Tensor) else antenna_imgs
        self.frequencies = torch.FloatTensor(frequencies) if not isinstance(frequencies, torch.Tensor) else frequencies
        self.distances = torch.FloatTensor(distances) if not isinstance(distances, torch.Tensor) else distances
        self.bs_heights = torch.FloatTensor(bs_heights) if not isinstance(bs_heights, torch.Tensor) else bs_heights
        self.app_service = torch.FloatTensor(app_service) if not isinstance(app_service, torch.Tensor) else app_service
        self.target_tx_power = torch.FloatTensor(target_tx_power) if not isinstance(target_tx_power, torch.Tensor) else target_tx_power
        
    def __len__(self):
        return len(self.target_tx_power)
    
    def __getitem__(self, idx):
        return (
            self.building_imgs[idx],
            self.antenna_imgs[idx],
            self.frequencies[idx],
            self.distances[idx],
            self.bs_heights[idx],
            self.app_service[idx],
            self.target_tx_power[idx]
        )

# 2 - Load data: images and csv files ===========================================================================================================

def load_images_from_folder(folder_path, num_samples=None, target_size=(256, 256)):

    # Select files that are images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    
    if num_samples is not None:
        image_files = image_files[:num_samples]
    
    print(f"Loading {len(image_files)} images from {folder_path}...")
    
    images = []
    for img_file in tqdm(image_files, desc="Loading images"):
        img_path = os.path.join(folder_path, img_file)
        # Load image and convert to grayscale
        img = Image.open(img_path).convert('L')
        # Resize to target size
        img = img.resize(target_size, Image.BILINEAR)
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        images.append(img_array)
    
    images = np.array(images)  # Shape: [num_samples, 256, 256]
    images = images[:, np.newaxis, :, :]  # Add channel dimension: [num_samples, 1, 256, 256]
    
    return images


def load_data(base_path, num_samples=None, train_ratio=0.8):

    # Define common folder for simplification
    base_path = Path(base_path)
    
    # Load building images
    buildings_folder = base_path / "building_maps_test"
    building_imgs = load_images_from_folder(str(buildings_folder), num_samples)
    print(f"Building images shape: {building_imgs.shape}")
    
    # Load antenna images
    antenna_folder = base_path / "antenna_maps_test"
    antenna_imgs = load_images_from_folder(str(antenna_folder), num_samples)
    print(f"Antenna images shape: {antenna_imgs.shape}")
    
    # Verify same number of samples
    assert building_imgs.shape[0] == antenna_imgs.shape[0], \
        "Building and antenna images must have same number of samples!"
    
    actual_num_samples = building_imgs.shape[0]
    
    print("\n")
    # Load min_distance_map CSV
    min_distance_df = pd.read_csv(base_path / "min_distance_map.csv")
    print(f"min_distance_map.csv loaded: {len(min_distance_df)} rows")
    
    # Load cartoradio_map CSV
    cartoradio_df = pd.read_csv(base_path / "df_sites_antennas.csv")
    print(f"cartoradio_map.csv loaded: {len(cartoradio_df)} rows")
    
    # Extract data from CSVs
    frequencies = cartoradio_df['Début'].values[:actual_num_samples, np.newaxis]
    distances = min_distance_df['Distance [km]'].values[:actual_num_samples, np.newaxis]
    bs_heights = cartoradio_df['Hauteur / sol'].values[:actual_num_samples, np.newaxis]
    app_service = min_distance_df['File type'].values[:actual_num_samples, np.newaxis]
    target_tx_power = min_distance_df['Total Tx Power mW'].values[:actual_num_samples, np.newaxis]
    
    # One hot encoding for application service type (since the value from table is a string)
    label_encoder = LabelEncoder()
    app_service_encoded = label_encoder.fit_transform(app_service)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    app_service = onehot_encoder.fit_transform(app_service_encoded.reshape(-1,1))

    # Prints for sanity check
    print(f"\nFrequencies shape: {frequencies.shape}")
    print(f"Distances shape: {distances.shape}")
    print(f"BS heights shape: {bs_heights.shape}")
    print(f"Application service shape: {app_service.shape}")
    print(f"Target TX power shape: {target_tx_power.shape}")
    
    # Split data into train and validation
    indices = np.arange(actual_num_samples)
    train_idx, val_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    
    print(f"\n{'='*60}")
    print(f"DATASET SPLIT")
    print(f"{'='*60}")
    print(f"Total samples: {actual_num_samples}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"{'='*60}\n")
    
    return {
        'train': {
            'building_imgs': building_imgs[train_idx],
            'antenna_imgs': antenna_imgs[train_idx],
            'frequencies': frequencies[train_idx],
            'distances': distances[train_idx],
            'bs_heights': bs_heights[train_idx],
            'app_service': app_service[train_idx],
            'target_tx_power': target_tx_power[train_idx]
        },
        'val': {
            'building_imgs': building_imgs[val_idx],
            'antenna_imgs': antenna_imgs[val_idx],
            'frequencies': frequencies[val_idx],
            'distances': distances[val_idx],
            'bs_heights': bs_heights[val_idx],
            'app_service': app_service[val_idx],
            'target_tx_power': target_tx_power[val_idx]
        }
    }

# 4 - Training ===========================================================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        building_imgs, antenna_imgs, frequencies, distances, bs_heights, app_service, target_tx_power = batch
        
        # Move to device
        building_imgs = building_imgs.to(device)
        antenna_imgs = antenna_imgs.to(device)
        frequencies = frequencies.to(device)
        distances = distances.to(device)
        bs_heights = bs_heights.to(device)
        app_service = app_service.to(device)
        target_tx_power = target_tx_power.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        tx_power, RSRP = model(building_imgs, antenna_imgs, frequencies, 
                               distances, bs_heights, app_service)
        
        # Calculate loss
        loss = criterion(tx_power, target_tx_power)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item() * building_imgs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            building_imgs, antenna_imgs, frequencies, distances, bs_heights, app_service, target_tx_power = batch
            
            # Move to device
            building_imgs = building_imgs.to(device)
            antenna_imgs = antenna_imgs.to(device)
            frequencies = frequencies.to(device)
            distances = distances.to(device)
            bs_heights = bs_heights.to(device)
            app_service = app_service.to(device)
            target_tx_power = target_tx_power.to(device)
            
            # Forward pass
            tx_power, RSRP = model(building_imgs, antenna_imgs, frequencies, 
                                   distances, bs_heights, app_service)
            
            # Calculate loss
            loss = criterion(tx_power, target_tx_power)
            
            running_loss += loss.item() * building_imgs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, num_epochs=100, device='cuda', save_path='best_model.pth'):
    """Complete training loop with validation"""
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"✓ Best model saved with val_loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(9, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()
    
    return train_losses, val_losses

# 5 - Evaluation and Plotting ==================================================================================================================

def evaluate_model(model, dataloader, device):
    """
    Evaluate model and collect predictions and targets
    
    Returns:
        predictions: numpy array of predicted Tx power values
        targets: numpy array of ground truth Tx power values
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            building_imgs, antenna_imgs, frequencies, distances, bs_heights, app_service, target_tx_power = batch
            
            # Move to device
            building_imgs = building_imgs.to(device)
            antenna_imgs = antenna_imgs.to(device)
            frequencies = frequencies.to(device)
            distances = distances.to(device)
            bs_heights = bs_heights.to(device)
            app_service = app_service.to(device)
            
            # Forward pass
            tx_power, _ = model(building_imgs, antenna_imgs, frequencies, 
                               distances, bs_heights, app_service)
            
            all_predictions.append(tx_power.cpu().numpy())
            all_targets.append(target_tx_power.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    return predictions, targets


def plot_predictions_vs_targets(predictions, targets, save_path='prediction_vs_target.png'):
    """
    Plot 1: Predicted vs Ground Truth with linear fit and R^2 (in dB scale)
    """
    from scipy import stats
    
    # Convert to dB scale (10 * log10(power_mW))
    predictions_dB = 10 * np.log10(predictions + 1e-10)  # Add small epsilon to avoid log(0)
    targets_dB = 10 * np.log10(targets + 1e-10)
    
    # Calculate linear regression in dB domain
    slope, intercept, r_value, p_value, std_err = stats.linregress(targets_dB, predictions_dB)
    r_squared = r_value ** 2

    # Calculate R^2 with respect to perfect predictino line
    ss_res_perfect = np.sum((predictions_dB - targets_dB) ** 2)  # residuals from y=x line
    ss_tot = np.sum((predictions_dB - np.mean(predictions_dB)) ** 2)  # total variance
    r_squared_perfect = 1 - (ss_res_perfect / ss_tot)  # R^2 vs perfect prediction   
    
    # Create fitted line
    line = slope * targets_dB + intercept
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(targets_dB, predictions_dB, alpha=0.5, s=20, label='Predictions')
    plt.plot(targets_dB, line, 'r-', linewidth=2, label=f'Linear Fit (R² = {r_squared:.4f})')
    plt.plot([targets_dB.min(), targets_dB.max()], [targets_dB.min(), targets_dB.max()], 
             'k--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Tx Power (dBm)', fontsize=14)
    plt.ylabel('Predicted Tx Power (dBm)', fontsize=14)
    plt.title('Predicted vs Ground Truth Tx Power (dB Scale)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = (f'N = {len(predictions)}\n'
               f'R² (vs fit) = {r_squared:.4f}\n'
               f'R² (vs y=x) = {r_squared_perfect:.4f}\n'
               f'Slope = {slope:.4f}\n'
               f'Intercept = {intercept:.4f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"PREDICTION STATISTICS (dB scale)")
    print(f"{'='*60}")
    print(f"Number of samples: {len(predictions)}")
    print(f"R² Score (vs regression line): {r_squared:.4f}")
    print(f"R² Score (vs perfect prediction): {r_squared_perfect:.4f}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(predictions_dB - targets_dB)):.4f} dB")
    print(f"{'='*60}\n")
    
    return r_squared, r_squared_perfect, slope, intercept, predictions_dB, targets_dB

def plot_error_histogram(predictions_dB, targets_dB, slope, intercept, save_path='error_histogram.png'):
    """
    Plot 2: Histogram of errors in dB (distance from predictions to linear fit)
    """
    # Calculate fitted values from linear regression (in dB)
    fitted_values = slope * targets_dB + intercept
    
    # Calculate errors (residuals) in dB
    errors = predictions_dB - fitted_values
    
    # Calculate statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Add vertical line at mean
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_error:.4f} dB')
    plt.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
    
    plt.xlabel('Error (Prediction - Fitted Value) (dB)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Prediction Errors (dB Scale)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Mean Error = {mean_error:.4f} dB\nStd Error = {std_error:.4f} dB'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"{'='*60}")
    print(f"ERROR STATISTICS (dB scale)")
    print(f"{'='*60}")
    print(f"Mean Error: {mean_error:.4f} dB")
    print(f"Std Error: {std_error:.4f} dB")
    print(f"{'='*60}\n")

def train_with_mae_tracking(model, train_loader, val_loader, criterion, optimizer, 
                            scheduler=None, num_epochs=100, device='cuda', save_path='best_model.pth'):
    """
    Plot 3: Training loop that also tracks MAE alongside MSE
    """
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_mae_losses = []
    val_mae_losses = []
    
    mae_criterion = nn.L1Loss()  # MAE loss
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # ========== Training ==========
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            building_imgs, antenna_imgs, frequencies, distances, bs_heights, app_service, target_tx_power = batch
            
            # Move to device
            building_imgs = building_imgs.to(device)
            antenna_imgs = antenna_imgs.to(device)
            frequencies = frequencies.to(device)
            distances = distances.to(device)
            bs_heights = bs_heights.to(device)
            app_service = app_service.to(device)
            target_tx_power = target_tx_power.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            tx_power, RSRP = model(building_imgs, antenna_imgs, frequencies, 
                                   distances, bs_heights, app_service)
            
            # Calculate losses
            loss = criterion(tx_power, target_tx_power)  # MSE for backprop
            mae_loss = mae_criterion(tx_power, target_tx_power)  # MAE for tracking
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * building_imgs.size(0)
            running_mae += mae_loss.item() * building_imgs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae = running_mae / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_mae_losses.append(epoch_mae)
        
        # ========== Validation ==========
        model.eval()
        running_loss = 0.0
        running_mae = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                building_imgs, antenna_imgs, frequencies, distances, bs_heights, app_service, target_tx_power = batch
                
                # Move to device
                building_imgs = building_imgs.to(device)
                antenna_imgs = antenna_imgs.to(device)
                frequencies = frequencies.to(device)
                distances = distances.to(device)
                bs_heights = bs_heights.to(device)
                app_service = app_service.to(device)
                target_tx_power = target_tx_power.to(device)
                
                # Forward pass
                tx_power, RSRP = model(building_imgs, antenna_imgs, frequencies, 
                                       distances, bs_heights, app_service)
                
                # Calculate losses
                loss = criterion(tx_power, target_tx_power)
                mae_loss = mae_criterion(tx_power, target_tx_power)
                
                running_loss += loss.item() * building_imgs.size(0)
                running_mae += mae_loss.item() * building_imgs.size(0)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_mae = running_mae / len(val_loader.dataset)
        val_losses.append(epoch_loss)
        val_mae_losses.append(epoch_mae)
        
        print(f"Train Loss (MSE): {train_losses[-1]:.4f} | Train MAE: {epoch_mae:.4f}")
        print(f"Val Loss (MSE): {epoch_loss:.4f} | Val MAE: {epoch_mae:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(epoch_loss)
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_loss,
                'val_mae': epoch_mae,
            }, save_path)
            print(f"✓ Best model saved with val_loss: {epoch_loss:.4f}, val_mae: {epoch_mae:.4f}")
    
    return train_losses, val_losses, train_mae_losses, val_mae_losses


def plot_mae_curves(train_mae_losses, val_mae_losses, save_path='mae_curves.png'):
    """
    Plot 3: MAE over epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_mae_losses, label='Train MAE', linewidth=2)
    plt.plot(val_mae_losses, label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MAE Loss', fontsize=14)
    plt.title('Mean Absolute Error over Training', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 5 - Main ==================================================================================================================

# Main training script
if __name__ == "__main__":

    # Path to the data directory
    BASE_DATA_PATH = "/Users/alvaroribas/Documents/University/TelecomSudParis/PIR/Notebooks"
    
    # Number of samples 
    NUM_SAMPLES = None  # None = use all available, or insert the exact number
    
    # Training parameters
    BATCH_SIZE = 5
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.80  # 80% train, 20% validation
    
    # Use MPS (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load data
    data = load_data(BASE_DATA_PATH, num_samples=NUM_SAMPLES, train_ratio=TRAIN_RATIO)
    
    # Create datasets
    train_dataset = TxPowerDataset(
        data['train']['building_imgs'],
        data['train']['antenna_imgs'],
        data['train']['frequencies'],
        data['train']['distances'],
        data['train']['bs_heights'],
        data['train']['app_service'],
        data['train']['target_tx_power']
    )
    
    val_dataset = TxPowerDataset(
        data['val']['building_imgs'],
        data['val']['antenna_imgs'],
        data['val']['frequencies'],
        data['val']['distances'],
        data['val']['bs_heights'],
        data['val']['app_service'],
        data['val']['target_tx_power']
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    model = PowerModel().to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
        )
    
    # Print model info
    print(f"\n{'='*60}")
    print(f"MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'='*60}\n")
    
    # Train the model WITH MAE TRACKING
    train_losses, val_losses, train_mae_losses, val_mae_losses = train_with_mae_tracking(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path='best_tx_power_model.pth'
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best validation loss (MSE): {min(val_losses):.4f}")
    print(f"Best validation MAE: {min(val_mae_losses):.4f}")
    print("="*60)
    
    # ========== GENERATE EVALUATION PLOTS ==========
    
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load('best_tx_power_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    val_predictions, val_targets = evaluate_model(model, val_loader, device)
    
    # Plot 1: Predictions vs Targets with R² (in dB)
    r_squared, r_squared_perfect, slope, intercept, val_predictions_dB, val_targets_dB = plot_predictions_vs_targets(
        val_predictions, val_targets, 
        save_path='prediction_vs_target.png'
    )
    
    # Plot 2: Error Histogram (in dB)
    plot_error_histogram(
        val_predictions_dB, val_targets_dB, slope, intercept,
        save_path='error_histogram.png'
    )
    
    # Plot 3: MAE Curves
    plot_mae_curves(
        train_mae_losses, val_mae_losses,
        save_path='mae_curves.png'
    )
    
    # Also plot MSE curves (original)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (MSE)', linewidth=2)
    plt.plot(val_losses, label='Validation Loss (MSE)', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.title('Mean Squared Error over Training', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mse_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print("  - best_tx_power_model.pth")
    print("  - prediction_vs_target.png")
    print("  - error_histogram.png")
    print("  - mae_curves.png")
    print("  - mse_curves.png")
    print("="*60)  
