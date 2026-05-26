import numpy as np

def predict_mosquito_position(times, observations, target_future_time=0.08):
    """
    칼만 필터와 다항식 외삽법을 사용하여 모기의 미래 위치를 예측합니다.
    
    :param times: 관측 시간 배열 (예: [-0.4, -0.36, ..., 0.0])
    :param observations: 3차원 관측 좌표 배열 (11 x 3 shape)
    :param target_future_time: 예측하고자 하는 미래 시간 (기본값 80ms = 0.08s)
    :return: 칼만 필터 예측 좌표, 다항식 기반 예측 좌표
    """
    
    dt = 0.04  # 측정 주기 (40ms)
    
    # ---------------------------------------------------------
    # 1. 칼만 필터 (Kalman Filter) 설정 (등속도 모델 기반)
    # 상태 벡터 X = [x, y, z, vx, vy, vz]^T
    # ---------------------------------------------------------
    X = np.zeros(6) 
    X[0:3] = observations[0] # 초기 위치를 첫 번째 관측치로 설정
    
    # 상태 천이 행렬 (State Transition Matrix)
    F = np.eye(6)
    for i in range(3):
        F[i, i+3] = dt
        
    # 관측 행렬 (Measurement Matrix) - 위치 정보만 측정됨
    H = np.zeros((3, 6))
    for i in range(3):
        H[i, i] = 1.0
        
    # 공분산 행렬 초기화
    P = np.eye(6) * 1000.0  # 초기 추정 오차 공분산
    Q = np.eye(6) * 0.1     # 프로세스 노이즈 (모기의 불규칙한 기동을 반영)
    R = np.eye(3) * 5.0     # 측정 노이즈 (LiDAR 센서의 오차 반영)

    # 11개의 관측치에 대해 필터 업데이트 (Update & Predict)
    filtered_positions = []
    
    for z in observations:
        # --- 예측 단계 (Predict) ---
        X_pred = F @ X
        P_pred = F @ P @ F.T + Q
        
        # --- 업데이트 단계 (Update) ---
        y = z - (H @ X_pred) # 잔차 (Residual)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S) # 칼만 이득 (Kalman Gain)
        
        # ASTEKF 적용 시, 여기서 페이딩 요소(Fading factor)를 P_pred에 곱하여
        # 최근 잔차(y)가 클 경우 K값을 높이는 로직이 추가됩니다.
        
        X = X_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred
        
        filtered_positions.append(X[0:3])
        
    filtered_positions = np.array(filtered_positions)

    # --- 80ms(0.08초) 후의 미래 위치 예측 (칼만 필터 기반) ---
    # 예측 시간만큼 상태를 앞으로 진행시킴
    steps_ahead = int(target_future_time / dt)
    X_future = np.copy(X)
    for _ in range(steps_ahead):
        X_future = F @ X_future
        
    kf_prediction = X_future[0:3]


    # ---------------------------------------------------------
    # 2. 다항식 보간/외삽법 (Polynomial Extrapolation)
    # LiDAR 노이즈에 민감하므로 필터링된 좌표를 기반으로 피팅
    # ---------------------------------------------------------
    poly_degree = 3 # 3차 다항식 (급격한 방향 전환을 부드럽게 피팅)
    
    poly_prediction = np.zeros(3)
    target_time_val = times[-1] + target_future_time # 0.0 + 0.08 = 0.08
    
    for axis in range(3):
        # x, y, z 각 축에 대해 시간에 따른 다항식 계수 계산
        coeffs = np.polyfit(times, filtered_positions[:, axis], poly_degree)
        poly_func = np.poly1d(coeffs)
        poly_prediction[axis] = poly_func(target_time_val)


    return kf_prediction, poly_prediction


# ==========================================
# 실행 및 테스트 코드
# ==========================================
if __name__ == "__main__":
    # 1. 시간 배열 생성 (-400ms ~ 0ms, 40ms 간격)
    time_steps = np.round(np.arange(-0.4, 0.01, 0.04), 3)
    
    # 2. 가상의 모기 비행 데이터 생성 (임의의 궤적 + 노이즈)
    np.random.seed(42)
    true_x = 10 * np.sin(time_steps * 5)
    true_y = 10 * np.cos(time_steps * 5)
    true_z = 5 * time_steps + 20
    
    # LiDAR 측정치라고 가정한 데이터 (노이즈 추가)
    observations = np.column_stack((true_x, true_y, true_z))
    observations += np.random.normal(0, 1.0, observations.shape)
    
    print("--- 최근 3개 LiDAR 관측 좌표 (노이즈 포함) ---")
    for t, obs in zip(time_steps[-3:], observations[-3:]):
        print(f"Time {t:5.2f}s : X={obs[0]:6.2f}, Y={obs[1]:6.2f}, Z={obs[2]:6.2f}")

    # 3. 80ms 후 위치 예측
    kf_pred, poly_pred = predict_mosquito_position(time_steps, observations, target_future_time=0.08)
    
    print("\n--- 80ms (+0.08s) 이후 예측 결과 ---")
    print(f"[1] 칼만 필터 예측 (직선 투영) : X={kf_pred[0]:6.2f}, Y={kf_pred[1]:6.2f}, Z={kf_pred[2]:6.2f}")
    print(f"[2] 다항식 보간 예측 (곡선 반영): X={poly_pred[0]:6.2f}, Y={poly_pred[1]:6.2f}, Z={poly_pred[2]:6.2f}")