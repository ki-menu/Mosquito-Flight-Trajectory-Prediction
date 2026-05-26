"""
Fine-tuning script for Mosquito Flight Trajectory GRU model.
Finetunes a pre-trained Wing Loss model using Gaussian Hit Loss + Wing Loss (Anchor).
"""
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import argparse

from args import Config, set_seed
from model import MosquitoGRU, MosquitoGRU_M2M, WingLoss, GaussianHitLoss
from dataset import MosquitoDataset


class FineTuneLoss(torch.nn.Module):
    """
    Fine-tuning loss supporting both:
    1) Combined: alpha * WingLoss + beta * GaussianHitLoss
    2) SoftHit:  alpha * EuclidLoss + beta * SoftHitLoss
    """
    def __init__(self, wing_w=0.02, wing_epsilon=0.005, 
                 sigma_beam=0.005, sigma_mosquito=0.002, 
                 alpha=1.0, beta=0.3, loss_type='softhit', softhit_beta=0.002):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.softhit_beta = softhit_beta
        
        # Combined Loss components
        self.wing_loss = WingLoss(w=wing_w, epsilon=wing_epsilon)
        self.hit_loss = GaussianHitLoss(sigma_beam=sigma_beam, sigma_mosquito=sigma_mosquito)

    def forward(self, pred, target):
        if self.loss_type == 'softhit':
            # Wing Loss as Base/Anchor
            l_wing = self.wing_loss(pred, target)
            
            # SoftHit Loss: sigmoid((d - 0.01) / beta)
            d = torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12)
            l_softhit = torch.sigmoid((d - 0.01) / self.softhit_beta).mean()
            
            total = self.alpha * l_wing + self.beta * l_softhit
            loss_dict = {
                'wing_loss': l_wing.item(),  # Logged as wing_loss/val_wing_loss in wandb
                'hit_loss': l_softhit.item(),  # Logged as hit_loss/val_hit_loss in wandb
                'total_loss': total.item()
            }
        else:
            l_wing = self.wing_loss(pred, target)
            l_hit = self.hit_loss(pred, target)
            total = self.alpha * l_wing + self.beta * l_hit
            
            loss_dict = {
                'wing_loss': l_wing.item(),
                'hit_loss': l_hit.item(),
                'total_loss': total.item()
            }
        return total, loss_dict


def compute_step(outputs, targets, criterion, model_mode):
    """Calculates loss and +80ms distance. Supports both M2M and M2O."""
    if model_mode == 'm2m':
        pred_40, pred_80 = outputs[:, :3], outputs[:, 3:]
        tgt_40, tgt_80 = targets[:, :3], targets[:, 3:]

        loss_80, loss_dict = criterion(pred_80, tgt_80)

        valid_40 = ~torch.isnan(tgt_40).any(dim=1)
        if valid_40.any():
            loss_40, _ = criterion(pred_40[valid_40], tgt_40[valid_40])
            loss = loss_80 + 0.5 * loss_40
        else:
            loss = loss_80
        dists = torch.norm(pred_80.detach() - tgt_80, dim=1)
    else:
        loss, loss_dict = criterion(outputs, targets)
        dists = torch.norm(outputs.detach() - targets, dim=1)

    return loss, dists, loss_dict


def main():
    parser = argparse.ArgumentParser(description="Mosquito Trajectory GRU Fine-Tuning")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to pre-trained pth model. If None, auto-detects the best model in ./result/")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Fine-tuning learning rate (default: 2e-5)")
    parser.add_argument('--epochs', type=int, default=40,
                        help="Fine-tuning epochs (default: 40)")
    parser.add_argument('--patience', type=int, default=4,
                        help="Scheduler patience (default: 4)")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Loss coefficient alpha (default: 1.0)")
    parser.add_argument('--beta', type=float, default=0.3,
                        help="Loss coefficient beta (default: 0.3)")
    parser.add_argument('--loss-type', type=str, default='softhit',
                        choices=['combined', 'softhit'],
                        help="Loss function combination style (default: softhit)")
    parser.add_argument('--softhit-beta', type=float, default=0.002,
                        help="Temperature parameter for SoftHit loss (default: 0.002)")
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'mps'],
                        help="Device selection")
    parser.add_argument('--subseq-min', type=int, default=4,
                        help="Sub-sequence augmentation minimum length (default: 4)")
    parser.add_argument('--no-subseq-aug', dest='subseq_aug', action='store_false',
                        help="Disable sub-sequence augmentation for training (default: Enabled)")
    parser.set_defaults(subseq_aug=True)
    args = parser.parse_args()

    # 1. Base Setup & Seed
    set_seed(Config.seed)
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    elif args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    Config.device = device
    Config.subseq_min_len = args.subseq_min
    print(f"Using device: {device}")

    # 2. Base Model Path Auto-Detection
    result_dir = Path('./result')
    if args.model_path is not None:
        base_model_path = Path(args.model_path)
    else:
        model_files = sorted(list(result_dir.glob('gru_*.pth')))
        if not model_files:
            raise FileNotFoundError("Error: No pre-trained gru_*.pth model found in ./result/ folder.")
        base_model_path = model_files[0]
        
    print(f"Base model selected for fine-tuning: {base_model_path}")

    # Load state dict and auto-detect model mode
    state_dict = torch.load(base_model_path, map_location='cpu')
    is_m2m = 'fc_40.weight' in state_dict
    model_mode = 'm2m' if is_m2m else 'm2o'
    Config.model_mode = model_mode
    Config.use_delta = True
    Config.use_rotation = True
    print(f"Detected Model Mode: {model_mode.upper()} (is_m2m = {is_m2m})")

    # 3. Model Initialization
    if is_m2m:
        model = MosquitoGRU_M2M(
            input_size=Config.input_size,
            hidden_size=Config.hidden_size,
            num_layers=Config.num_layers,
            dropout_rate=Config.dropout_rate
        ).to(device)
    else:
        model = MosquitoGRU(
            input_size=Config.input_size,
            hidden_size=Config.hidden_size,
            num_layers=Config.num_layers,
            output_size=Config.output_size,
            dropout_rate=Config.dropout_rate
        ).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    print("Pre-trained weights loaded successfully.")

    # 4. WandB Init
    wandb.init(
        project="DACON-2605-Mosquito-Trajectory",
        name=f"{Config.run_name}_FineTune_Gaussian",
        config={
            "stage": "Fine-Tuning",
            "base_model": base_model_path.name,
            "model_mode": Config.model_mode,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "loss_type": args.loss_type,
            "alpha": args.alpha,
            "beta": args.beta,
            "softhit_beta": args.softhit_beta,
            "sigma_beam": Config.sigma_beam,
            "sigma_mosquito": Config.sigma_mosquito
        }
    )

    # 5. Dataset Loading (with exact splits as train.py)
    train_files = sorted(list(Config.train_dir.glob('TRAIN_*.csv')))
    train_labels = pd.read_csv(Config.train_labels_path)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=Config.seed)

    print("Loading datasets...")
    train_dataset = MosquitoDataset(train_files, train_labels, is_train=True,
                                    use_delta=Config.use_delta,
                                    use_rotation=Config.use_rotation,
                                    subseq_aug=args.subseq_aug,
                                    subseq_min_len=Config.subseq_min_len,
                                    subseq_max_len=Config.subseq_max_len,
                                    model_mode=Config.model_mode)
    
    val_dataset = MosquitoDataset(val_files, train_labels, is_train=True,
                                  use_delta=Config.use_delta,
                                  use_rotation=Config.use_rotation,
                                  subseq_aug=False,
                                  model_mode=Config.model_mode)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    print(f"Datasets loaded! Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 6. Loss, Optimizer, and Scheduler
    criterion = FineTuneLoss(
        wing_w=Config.wing_w, wing_epsilon=Config.wing_epsilon,
        sigma_beam=Config.sigma_beam, sigma_mosquito=Config.sigma_mosquito,
        alpha=args.alpha, beta=args.beta,
        loss_type=args.loss_type, softhit_beta=args.softhit_beta
    ).to(device)

    # Lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=Config.scheduler_factor, 
        patience=args.patience,
        min_lr=Config.min_lr
    )

    # 7. Fine-Tuning Loop
    ACC_THRESHOLD = 0.01  # 1cm
    best_val_loss = float('inf')
    best_val_dist_total = float('inf')
    best_epoch = 0

    print("\n--- Starting Fine-Tuning Stage ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_dist = 0.0
        train_correct = 0
        train_wing_loss = 0.0
        train_hit_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] FT Train", leave=False)
        for seq, target in train_pbar:
            seq, target = seq.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(seq)
            loss, dists, loss_dict = compute_step(outputs, target, criterion, Config.model_mode)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seq.size(0)
            train_dist += dists.sum().item()
            train_correct += (dists < ACC_THRESHOLD).sum().item()
            
            train_wing_loss += loss_dict.get('wing_loss', 0.0) * seq.size(0)
            train_hit_loss += loss_dict.get('hit_loss', 0.0) * seq.size(0)
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_dist /= n_train
        train_acc = train_correct / n_train
        train_wing_loss /= n_train
        train_hit_loss /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        val_correct = 0
        val_wing_loss = 0.0
        val_hit_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] FT Val", leave=False)
        
        with torch.no_grad():
            for seq, target in val_pbar:
                seq, target = seq.to(device), target.to(device)
                outputs = model(seq)
                loss, dists, loss_dict_val = compute_step(outputs, target, criterion, Config.model_mode)
                val_loss += loss.item() * seq.size(0)
                val_dist += dists.sum().item()
                val_correct += (dists < ACC_THRESHOLD).sum().item()
                
                val_wing_loss += loss_dict_val.get('wing_loss', 0.0) * seq.size(0)
                val_hit_loss += loss_dict_val.get('hit_loss', 0.0) * seq.size(0)

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_dist /= n_val
        val_acc = val_correct / n_val
        val_wing_loss /= n_val
        val_hit_loss /= n_val

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f} | "
              f"Train Dist: {train_dist:.4f}  Val Dist: {val_dist:.4f} | "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/dist": train_dist,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/dist": val_dist,
            "val/acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "train/wing_loss": train_wing_loss,
            "train/hit_loss": train_hit_loss,
            "val/wing_loss": val_wing_loss,
            "val/hit_loss": val_hit_loss
        }
        wandb.log(log_dict)

        scheduler.step(val_loss)

        # Save model on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dist_total = val_dist
            best_epoch = epoch + 1
            
            torch.save(model.state_dict(), result_dir / 'best_model_ft_tmp.pth')
            print(f"  --> Updated best fine-tuned model (Epoch {best_epoch}, Loss: {best_val_loss:.6f}, Dist: {best_val_dist_total:.4f})")
            
            wandb.summary["best_ft_val_loss"] = best_val_loss
            wandb.summary["best_ft_val_dist"] = best_val_dist_total
            wandb.summary["best_ft_val_acc"] = val_acc
            wandb.summary["best_ft_epoch"] = best_epoch

    # Rename final best checkpoint
    if best_epoch > 0:
        final_model_path = result_dir / f'gru_finetuned_{best_val_dist_total:.4f}_{best_epoch}.pth'
        if os.path.exists(result_dir / 'best_model_ft_tmp.pth'):
            os.rename(result_dir / 'best_model_ft_tmp.pth', final_model_path)
            print(f"\nFine-Tuning complete. Best fine-tuned model saved to: {final_model_path}")

    wandb.finish()

if __name__ == '__main__':
    main()
