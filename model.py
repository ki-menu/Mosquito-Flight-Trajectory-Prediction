"""
Model definitions for Mosquito Flight Trajectory prediction.
"""
import math
import torch
import torch.nn as nn


class MosquitoGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3, dropout_rate=0.2):
        super(MosquitoGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어 (num_layers > 1 일때 레이어 사이에 dropout 적용)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out


class MosquitoGRU_M2M(nn.Module):
    """GRU predicting both +40ms and +80ms future positions.

    Output: (B, 6) = [pred_40ms (3D), pred_80ms (3D)].
    For inference/submission use the last 3 values (pred_80ms).
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_40 = nn.Linear(hidden_size, 3)
        self.fc_80 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # x: (B, T, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return torch.cat([self.fc_40(out), self.fc_80(out)], dim=1)  # (B, 6)


class WingLoss(nn.Module):
    def __init__(self, w=0.05, epsilon=0.01):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, y_pred, y_true):
        x = torch.abs(y_pred - y_true)
        loss = torch.where(
            x < self.w,
            self.w * torch.log(1.0 + x / self.epsilon),
            x - self.c
        )
        return loss.mean()


class GaussianHitLoss(nn.Module):
    """3D Gaussian-based Hit Loss for laser-mosquito engagement.

    두 등방(Isotropic) 3D 가우시안 분포의 overlap ratio를 closed-form으로 계산.
    - G_beam: 예측 좌표(μ_pred)를 중심으로 한 레이저 빔의 에너지 분포
    - M_vol:  정답 좌표(μ_true)를 중심으로 한 모기의 3D 부피

    Loss = 1 - overlap_ratio

    overlap = (σ_b² / (σ_b² + σ_m²))^(3/2) · exp(-||μ_pred - μ_true||² / (2(σ_b² + σ_m²)))

    Args:
        sigma_beam: 레이저 빔(타격 판정)의 유효 표준편차 (m).
                    대회 정답 인정 거리 0.01m 기준으로 설정.
        sigma_mosquito: 모기 유효 크기의 표준편차 (m).
    """
    def __init__(self, sigma_beam=0.005, sigma_mosquito=0.002):
        super().__init__()
        self.sigma_beam = sigma_beam
        self.sigma_mosquito = sigma_mosquito

        # 사전 계산 가능한 상수
        sigma_sum_sq = sigma_beam ** 2 + sigma_mosquito ** 2
        self.register_buffer(
            '_inv_2sigma_sum_sq',
            torch.tensor(1.0 / (2.0 * sigma_sum_sq))
        )
        self.register_buffer(
            '_scale',
            torch.tensor((sigma_beam ** 2 / sigma_sum_sq) ** 1.5)
        )

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 3) 모델 예측 좌표 (변환된 좌표계).
            target: (B, 3) 정답 좌표 (변환된 좌표계).

        Returns:
            스칼라 손실값.
        """
        diff = pred - target                                    # (B, 3)
        dist_sq = (diff ** 2).sum(dim=-1)                       # (B,)

        overlap = self._scale * torch.exp(-dist_sq * self._inv_2sigma_sum_sq)  # (B,)

        return (1.0 - overlap).mean()


class CombinedLoss(nn.Module):
    """WingLoss + GaussianHitLoss를 Curriculum Learning 전략으로 결합.

    학습 진행률(progress = epoch / total_epochs)에 따라 가중치를 동적 조절:
      - 초반 (progress < transition_start):
          alpha = 1.0,   beta = beta_min  → WingLoss 위주로 빠르게 수렴
      - 전환 구간 (transition_start ~ transition_end):
          alpha: 1.0 → alpha_min,  beta: beta_min → 1.0  (선형 보간)
      - 후반 (progress > transition_end):
          alpha = alpha_min,  beta = 1.0  → HitLoss 위주로 명중률 극대화

    Total Loss = alpha · WingLoss + beta · HitLoss

    Args:
        wing_w, wing_epsilon: WingLoss 파라미터.
        sigma_beam, sigma_mosquito: GaussianHitLoss 파라미터.
        transition_start: curriculum 전환 시작 비율 (0~1).
        transition_end: curriculum 전환 완료 비율 (0~1).
        alpha_min: 전환 완료 후 WingLoss 최소 가중치.
        beta_min: 학습 초기 HitLoss 최소 가중치.
    """
    def __init__(self, wing_w=0.03, wing_epsilon=0.005,
                 sigma_beam=0.008, sigma_mosquito=0.003,
                 transition_start=0.3, transition_end=0.7,
                 alpha_min=0.3, beta_min=0.05):
        super().__init__()
        self.wing_loss = WingLoss(w=wing_w, epsilon=wing_epsilon)
        self.hit_loss = GaussianHitLoss(sigma_beam=sigma_beam,
                                         sigma_mosquito=sigma_mosquito)

        self.transition_start = transition_start
        self.transition_end = transition_end
        self.alpha_min = alpha_min
        self.beta_min = beta_min

        # 현재 가중치 (set_progress로 갱신)
        self._alpha = 1.0
        self._beta = beta_min

    def set_progress(self, progress):
        """학습 진행률(0~1)에 따른 가중치 갱신.

        Args:
            progress: epoch / total_epochs (0.0 ~ 1.0).
        """
        if progress < self.transition_start:
            self._alpha = 1.0
            self._beta = self.beta_min
        elif progress > self.transition_end:
            self._alpha = self.alpha_min
            self._beta = 1.0
        else:
            # 선형 보간
            t = (progress - self.transition_start) / (self.transition_end - self.transition_start)
            self._alpha = 1.0 + t * (self.alpha_min - 1.0)
            self._beta = self.beta_min + t * (1.0 - self.beta_min)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 3) 모델 예측.
            target: (B, 3) 정답.

        Returns:
            total_loss: 스칼라 손실.
            loss_dict:  개별 항 값 (로깅용).
        """
        l_wing = self.wing_loss(pred, target)
        l_hit = self.hit_loss(pred, target)

        total = self._alpha * l_wing + self._beta * l_hit

        loss_dict = {
            'wing_loss': l_wing.item(),
            'hit_loss': l_hit.item(),
            'alpha': self._alpha,
            'beta': self._beta,
        }
        return total, loss_dict


class PhysicsInformedLoss(nn.Module):
    """WingLoss + 물리학적 역학 제약 기반 정규화 항을 결합한 복합 손실 함수.

    개별 물리 손실 클래스를 조합하여 사용한다.

    Args:
        w, epsilon: WingLoss 파라미터.
        use_delta: True이면 입력이 delta(변위 벡터) 모드.
        dt: 타임스텝 간격 (초). 기본 0.040 (40ms).
        pred_horizon: 예측 시점까지의 시간 (초). 기본 0.080 (80ms).
        d_max: 속도 제한 패널티의 최대 허용 변위 (m). None이면 자동 추정.
        lambda_vel: 속도 제한 패널티 가중치.
        lambda_inertia: 관성 패널티 가중치.
        lambda_dir: 방향 일관성 패널티 가중치.
        lambda_speed: 속도 크기 일관성 패널티 가중치.
    """
    def __init__(self, w=0.02, epsilon=0.005,
                 use_delta=True, dt=0.040, pred_horizon=0.080,
                 d_max=None,
                 lambda_vel=0.1,
                 lambda_inertia=0.05,
                 lambda_dir=0.05,
                 lambda_speed=0.02):
        super().__init__()
        self.use_delta = use_delta
        self.dt = dt
        self.pred_horizon = pred_horizon

        # 개별 손실 클래스 조합
        self.wing    = WingLoss(w=w, epsilon=epsilon)
        self.vel     = VelocityHingeLoss(d_max=d_max, pred_horizon=pred_horizon)
        self.inertia = KinematicInertiaLoss(pred_horizon=pred_horizon)
        self.dir     = DirectionalCosineLoss()
        self.speed   = SpeedConsistencyLoss(pred_horizon=pred_horizon)

        # 가중치
        self.lambda_vel     = lambda_vel
        self.lambda_inertia = lambda_inertia
        self.lambda_dir     = lambda_dir
        self.lambda_speed   = lambda_speed

    def forward(self, y_pred, y_true, seq):
        """물리 기반 손실을 계산한다.

        Args:
            y_pred: (B, 3) 모델 예측 (회전 좌표계에서의 변위).
            y_true: (B, 3) 정답 (회전 좌표계에서의 변위).
            seq: (B, T, 3) 입력 시퀀스 (정규화된 좌표 또는 변위).

        Returns:
            total_loss: 스칼라 손실.
            loss_dict: 각 항의 값을 담은 딕셔너리 (로깅용).
        """
        # 입력으로부터 운동학적 정보 추출
        v0 = extract_last_velocity(seq, self.use_delta, self.dt)
        avg_speed = extract_avg_speed(seq, self.use_delta, self.dt)

        # 각 손실 계산
        l_wing    = self.wing(y_pred, y_true)
        l_vel     = self.vel(y_pred)
        l_inertia = self.inertia(y_pred, v0)
        l_dir     = self.dir(y_pred, v0)
        l_speed   = self.speed(y_pred, avg_speed)

        total = (l_wing
                 + self.lambda_vel     * l_vel
                 + self.lambda_inertia * l_inertia
                 + self.lambda_dir     * l_dir
                 + self.lambda_speed   * l_speed)

        loss_dict = {
            'wing':      l_wing.item(),
            'vel_hinge': l_vel.item(),
            'inertia':   l_inertia.item(),
            'direction': l_dir.item(),
            'speed':     l_speed.item(),
        }

        return total, loss_dict


# ============================================================
# 개별 물리 손실 클래스
# ============================================================

class VelocityHingeLoss(nn.Module):
    """1. 속도 제한 패널티 (Velocity Hinge Loss).

    예측 변위가 물리적 한계 D_max를 초과하면 패널티를 부여한다.
    L_vel = mean(max(0, ||pred||₂ - D_max)²)

    모기의 최대 비행 속도(~2 m/s)를 기준으로, 주어진 시간 동안
    이동 가능한 최대 거리를 넘는 예측을 억제한다.
    """
    def __init__(self, d_max=None, pred_horizon=0.080):
        super().__init__()
        # D_max: 모기 최대 비행 속도 ~2.5 m/s 기준, 80ms 동안 이동 가능 거리
        self.d_max = d_max if d_max is not None else 2.5 * pred_horizon  # ~0.2m

    def forward(self, pred):
        """
        Args:
            pred: (B, 3) 모델 예측 변위.
        Returns:
            스칼라 손실.
        """
        pred_dist = torch.norm(pred, dim=-1)                # (B,)
        excess = torch.clamp(pred_dist - self.d_max, min=0)  # (B,)
        return (excess ** 2).mean()


class KinematicInertiaLoss(nn.Module):
    """2. 관성/가속도 제약 (Kinematic Inertia Loss).

    등속 직선 운동 가정 하의 예상 위치와 모델 예측 간의 차이를 패널티로 준다.
    P_inertia = V₀ × pred_horizon
    L_inertia = ||pred - P_inertia||₂²

    모기가 질량을 가진 물체로서 관성을 따르는 성질을 반영한다.
    """
    def __init__(self, pred_horizon=0.080):
        super().__init__()
        self.pred_horizon = pred_horizon

    def forward(self, pred, v0):
        """
        Args:
            pred: (B, 3) 모델 예측 변위.
            v0: (B, 3) 현재 속도 벡터 (m/s).
        Returns:
            스칼라 손실.
        """
        p_inertia = v0 * self.pred_horizon  # (B, 3) 등속 예상 변위
        return ((pred - p_inertia) ** 2).sum(dim=-1).mean()


class DirectionalCosineLoss(nn.Module):
    """3. 방향 일관성 (Directional Cosine Loss).

    과거 비행 방향과 예측 방향의 코사인 유사도를 기반으로 패널티.
    방향이 급격히 꺾이면 패널티가 커진다.
    L_dir = mean(1 - cos_sim(v_past, v_pred))

    속도가 매우 작은 경우(거의 정지)는 방향이 무의미하므로 제외.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, v0):
        """
        Args:
            pred: (B, 3) 모델 예측 변위 (방향 벡터로 사용).
            v0: (B, 3) 과거 속도 벡터.
        Returns:
            스칼라 손실.
        """
        v_past_norm = torch.norm(v0,   dim=-1, keepdim=True)  # (B, 1)
        v_pred_norm = torch.norm(pred, dim=-1, keepdim=True)  # (B, 1)

        # 속도가 극히 작은 샘플은 마스킹 (정지 상태 → 방향 무의미)
        valid = ((v_past_norm.squeeze(-1) > self.eps)
                 & (v_pred_norm.squeeze(-1) > self.eps))

        cos_sim = ((v0 * pred).sum(dim=-1)
                   / (v_past_norm.squeeze(-1) * v_pred_norm.squeeze(-1) + self.eps))
        loss = 1.0 - cos_sim  # (B,)  범위: [0, 2]

        if valid.any():
            return loss[valid].mean()
        return torch.tensor(0.0, device=pred.device)


class SpeedConsistencyLoss(nn.Module):
    """4. 속도 크기 일관성 (Speed Consistency Loss).

    예측 속력이 최근 평균 속력에서 크게 벗어나는 것을 억제한다.
    L_speed = mean((v_pred - v_avg)² / (v_avg² + eps))

    정규화된 상대 오차를 사용하여 빠른/느린 모기 모두에 공정하게 적용.
    """
    def __init__(self, pred_horizon=0.080, eps=1e-8):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.eps = eps

    def forward(self, pred, avg_speed):
        """
        Args:
            pred: (B, 3) 모델 예측 변위.
            avg_speed: (B,) 입력 시퀀스의 평균 속력 (m/s).
        Returns:
            스칼라 손실.
        """
        pred_speed = torch.norm(pred, dim=-1) / self.pred_horizon  # (B,)
        relative_error = (pred_speed - avg_speed) ** 2 / (avg_speed ** 2 + self.eps)
        return relative_error.mean()


# ============================================================
# 유틸리티: 시퀀스로부터 운동학적 정보 추출
# ============================================================

def extract_last_velocity(seq, use_delta, dt=0.040):
    """입력 시퀀스에서 마지막 스텝의 속도 벡터를 추출한다.

    Args:
        seq: (B, T, 3) 입력 시퀀스 (raw 또는 delta 모드).
        use_delta: delta 모드 여부.
        dt: 타임스텝 간격 (초).

    Returns:
        v0: (B, 3) 현재 속도 벡터 (m/s).
    """
    if use_delta:
        # delta 모드: seq[:, -1] 자체가 마지막 변위 벡터
        return seq[:, -1, :] / dt
    else:
        # raw 모드: 원점 이동 후이므로 seq[:, -1] = (0,0,0)
        # v0 = (P_0 - P_{-40}) / dt = (0 - seq[:, -2]) / dt
        return -seq[:, -2, :] / dt


def extract_avg_speed(seq, use_delta, dt=0.040):
    """입력 시퀀스에서 평균 속력(스칼라)을 계산한다.

    Args:
        seq: (B, T, 3)
        use_delta: delta 모드 여부.
        dt: 타임스텝 간격 (초).

    Returns:
        avg_speed: (B,) 평균 속력 (m/s).
    """
    if use_delta:
        # delta 모드: 각 스텝이 변위 벡터
        speeds = torch.norm(seq, dim=-1) / dt       # (B, T)
    else:
        # raw 모드: 위치 차분으로 속력 계산
        diffs = seq[:, 1:, :] - seq[:, :-1, :]      # (B, T-1, 3)
        speeds = torch.norm(diffs, dim=-1) / dt     # (B, T-1)
    return speeds.mean(dim=-1)                      # (B,)

