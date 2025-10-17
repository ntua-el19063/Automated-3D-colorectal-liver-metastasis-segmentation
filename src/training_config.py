from monai.networks.nets import UNet, BasicUNet, BasicUNetPlusPlus, UNETR, SegResNet, AttentionUnet, SwinUNETR
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss, HausdorffDTLoss
from torch.nn import CrossEntropyLoss

import torch, os, wandb
from torch.amp import GradScaler, autocast
from preprocessing import prepare
from utilities import train

# =========================
# GLOBAL CONFIG
# =========================
SPATIAL_SIZE = (112, 112, 64)     # used for cropping & sliding-window ROI
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 2
PATIENCE = 20
NUM_SAMPLES = 8
OVERLAP = 0.4
MODEL_NAME = "SegResNet"  # choose from NETWORKS dict
LOAD_CACHE = 0              


in_dir = r'path/to/data_split'
model_dir = r'path/to/results' #Store final model weights and visualizations
cache_path = r'path/to/cache.pt' #Save preprocessed data



# Networks dict

NETWORKS = {
    "UNet": lambda: UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=((2,2,1),(2,2,1),(2,2,2),(2,2,2)),
        num_res_units=2,
        act='PRELU', 
        norm='INSTANCE',
    ),
    "AttentionUnet": lambda: AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=((2,2,1),(2,2,1),(2,2,2),(2,2,2)),
        dropout=0.1,
    ),
    "SegResNet": lambda: SegResNet(
        spatial_dims=3,
        init_filters=16,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.1,
    ),
    "SwinUNETR": lambda: SwinUNETR(
        img_size=SPATIAL_SIZE,
        in_channels=1,
        out_channels=2,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.1,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    ),
}

# Data: cache or preprocess

if LOAD_CACHE:
    if os.path.exists(cache_path):
        print("Loading cached data...")
        train_loader, val_loader, test_loader = torch.load(cache_path)
    else:
        print(f"⚠️ Cache file not found at {cache_path}. Exiting.")
        exit(1)
else:
    if os.path.exists(cache_path):
        print("Overwriting previous data")
    print("Preprocessing data...")
    # NOTE: prepare must accept spatial_size; add it inside prepare() if needed
    train_loader, val_loader, test_loader = prepare(in_dir, cache=True, spatial_size=SPATIAL_SIZE, num_samples=NUM_SAMPLES)
    torch.save((train_loader, val_loader, test_loader), cache_path)


#print(f"Training set size: {len(train_loader)}")
#print(f"Validation set size: {len(val_loader)}")
#print(f"Test set size: {len(test_loader)}")

device = torch.device("cuda:0")
print(f"Using device: {device}")
print(torch.cuda.is_available())


loss_function = DiceFocalLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    gamma=2.0,
    weight= torch.tensor([1.0, 10.0], device=device),  # Lower weight for background, higher for tumor
    lambda_dice=1.0,
    lambda_focal=1.0,
)

# Train

if __name__ == '__main__':
    if MODEL_NAME not in NETWORKS:
        raise ValueError(f"MODEL_NAME must be one of {list(NETWORKS.keys())}, got: {MODEL_NAME}")

    print(f"\nTraining with {MODEL_NAME}")
    model = NETWORKS[MODEL_NAME]().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-6
    )

    scaler = GradScaler("cuda")

    network_model_dir = os.path.join(model_dir, MODEL_NAME)
    os.makedirs(network_model_dir, exist_ok=True)

    #WANDB for logs
    wandb.init(
        project="liver-tumor-seg",
        mode="online",
        name=f"SANITY - 2-IN | {MODEL_NAME} ",
        config={
            "architecture": MODEL_NAME,
            "activation_function": "softmax",
            "dataset": "optimized",
            "num_samples": NUM_SAMPLES,
            "spatial_size": f"{SPATIAL_SIZE}",
            "hardware_GPU": " ",
            "loss_weights": "1/10",
            "loss": "DiceFocal",
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "metric":"DiceMetric",
            "notes": " ",
        }
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss=loss_function,
        optim=optimizer,
        max_epochs=MAX_EPOCHS,
        model_dir=network_model_dir,
        patience=PATIENCE,
        spatial_size=SPATIAL_SIZE,
        overlap=OVERLAP,
        scheduler=None,
        scaler=scaler,
    )

