"""
Training loop for Mosquito Flight Trajectory GRU model.
"""
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

from args import Config, set_seed, parse_args, apply_args
from model import MosquitoGRU, WingLoss
from dataset import MosquitoDataset


def train():
    set_seed(Config.seed)
    print(f"Using device: {Config.device}")

    wandb.init(
        project="DACON-2605-Mosquito-Trajectory",
        name=Config.run_name,
        config={
            "model":        "GRU",
            "input_size":   Config.input_size,
            "hidden_size":  Config.hidden_size,
            "num_layers":   Config.num_layers,
            "dropout_rate": Config.dropout_rate,
            "output_size":  Config.output_size,
            "batch_size":   Config.batch_size,
            "epochs":       Config.epochs,
            "lr":           Config.lr,
            "device":       str(Config.device),
            "use_delta":    Config.use_delta,
            "use_rotation": Config.use_rotation,
            "patience":     Config.patience,
        },
    )

    # 파일 및 라벨 불러오기
    train_files = sorted(list(Config.train_dir.glob('TRAIN_*.csv')))
    train_labels = pd.read_csv(Config.train_labels_path)
    
    # 검증셋 분리 (8:2)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=Config.seed)
    
    # 학습에 적용할 augmentation 함수 목록 (원하는 함수를 추가/제거)
    augment_fns = [
        # translate_last_to_origin,
    ]

    # 데이터 로더 생성 (검증셋은 augmentation 없이)
    train_dataset = MosquitoDataset(train_files, train_labels, is_train=True,
                                    augment_fns=augment_fns,
                                    use_delta=Config.use_delta,
                                    use_rotation=Config.use_rotation,
                                    subseq_aug=True,
                                    subseq_min_len=Config.subseq_min_len,
                                    subseq_max_len=Config.subseq_max_len)
    val_dataset   = MosquitoDataset(val_files, train_labels, is_train=True,
                                    use_delta=Config.use_delta,
                                    use_rotation=Config.use_rotation,
                                    subseq_aug=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 모델, 손실함수, 옵티마이저 초기화
    model = MosquitoGRU(
        input_size=Config.input_size, 
        hidden_size=Config.hidden_size, 
        num_layers=Config.num_layers,
        output_size=Config.output_size,
        dropout_rate=Config.dropout_rate
    ).to(Config.device)
    
    # 모델 가중치 및 기울기 로그 기록 설정
    wandb.watch(model, log='all', log_freq=100)
    
    criterion = WingLoss(w=0.03, epsilon=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=Config.scheduler_factor, 
        patience=Config.patience,
        min_lr=Config.min_lr
    )
    
    ACC_THRESHOLD = 0.01  # 정답 인정 거리 기준 (m)
    best_val_loss = float('inf')
    best_val_dist_total = float('inf')
    best_epoch = 0

    Config.output_dir.mkdir(exist_ok=True)

    for epoch in range(Config.epochs):
        # ── 1. Warm-up 전략 적용 ──────────────────────────────────────
        if epoch < Config.warmup_epochs:
            curr_lr = Config.lr * (epoch + 1) / Config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
        
        model.train()
        train_loss = 0.0
        train_dist = 0.0
        train_correct = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.epochs}] Train", leave=False)
        for seq, target in train_pbar:
            seq, target = seq.to(Config.device), target.to(Config.device)

            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            dists = torch.norm(outputs.detach() - target, dim=1)
            train_loss += loss.item() * seq.size(0)
            train_dist += dists.sum().item()
            train_correct += (dists < ACC_THRESHOLD).sum().item()
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_dist /= n_train
        train_acc = train_correct / n_train

        # 검증
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        val_correct = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{Config.epochs}] Val", leave=False)
        with torch.no_grad():
            for seq, target in val_pbar:
                seq, target = seq.to(Config.device), target.to(Config.device)
                outputs = model(seq)
                loss = criterion(outputs, target)
                dists = torch.norm(outputs - target, dim=1)
                val_loss += loss.item() * seq.size(0)
                val_dist += dists.sum().item()
                val_correct += (dists < ACC_THRESHOLD).sum().item()

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_dist /= n_val
        val_acc = val_correct / n_val

        print(f"Epoch [{epoch+1}/{Config.epochs}] "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f} | "
              f"Train Dist: {train_dist:.4f}  Val Dist: {val_dist:.4f} | "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        wandb.log({
            "epoch":         epoch + 1,
            "train/loss":    train_loss,
            "train/dist":    train_dist,
            "train/acc":     train_acc,
            "val/loss":      val_loss,
            "val/dist":      val_dist,
            "val/acc":       val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        # 스케줄러 업데이트 (Warm-up 이후에만 작동하도록 설정 가능)
        if epoch >= Config.warmup_epochs:
            scheduler.step(val_loss)

        # 성능이 개선되면 모델 저장 (임시 저장)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dist_total = val_dist
            best_epoch = epoch + 1
            
            # 중간에 꺼질 수 있으므로 임시 파일로 계속 갱신
            torch.save(model.state_dict(), Config.output_dir / 'best_model_tmp.pth')
            
            print(f"  --> Updated best model (Epoch {best_epoch}, Loss: {best_val_loss:.6f})")
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_val_dist"] = best_val_dist_total
            wandb.summary["best_val_acc"]  = val_acc
            wandb.summary["best_epoch"]    = best_epoch
            wandb.summary["patience"]      = Config.patience

    # 학습 종료 후 최종 파일명으로 변경 (Dist값과 에폭 포함)
    if best_epoch > 0:
        final_model_path = Config.output_dir / f'gru_{best_val_dist_total:.4f}_{best_epoch}.pth'
        os.rename(Config.output_dir / 'best_model_tmp.pth', final_model_path)
        print(f"\nTraining complete. Final best model saved to: {final_model_path}")

    wandb.finish()
    return best_val_dist_total, best_epoch


if __name__ == '__main__':
    args = parse_args()
    apply_args(args)

    print("--- Starting GRU Training ---")
    train()
