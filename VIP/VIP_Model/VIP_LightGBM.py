import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# 경고 무시
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    CSV 데이터를 불러와 기본적인 정보를 출력합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. ({file_path})")
        return None

    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"데이터 로드 완료: {df.shape}")
        return df
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return None

def preprocess_initial(df):
    """
    1. 데이터 정렬 (발급회원번호, 기준년월)
    2. 이용금액 파생변수 생성 (당월 총 이용금액, 직전 3개월 평균)
    """
    df = df.sort_values(by=['발급회원번호', '기준년월']).copy()

    # 당월 총 이용금액
    if '이용금액_신용_B0M' in df.columns and '이용금액_체크_B0M' in df.columns:
        df['당월_총_이용금액'] = df['이용금액_신용_B0M'] + df['이용금액_체크_B0M']
    
    # 직전 3개월 평균 이용금액
    if '이용금액_신용_R3M' in df.columns and '이용금액_체크_R3M' in df.columns:
        df['직전_3M_평균_이용금액'] = (df['이용금액_신용_R3M'] + df['이용금액_체크_R3M']) / 3

    return df

def _define_churn_strict_logic(row):
    """
    행 단위 이탈 로직 (내부 함수)
    """
    # 직전 3개월 평균이 없거나 0원인 경우 -> '판단 제외'
    if pd.isna(row.get('직전_3M_평균_이용금액')) or row.get('직전_3M_평균_이용금액') <= 0:
        return np.nan

    # 당월 이용금액이 직전 평균의 80% 미만이면 이탈(1)
    if row.get('당월_총_이용금액') < (row.get('직전_3M_평균_이용금액') * 0.8):
        return 1
    else:
        return 0

def define_target(df):
    """
    이탈 타겟(Method B strict)을 정의하고 학습용/제외용 데이터로 분리합니다.
    """
    print("타겟 정의 중 (Method B strict)...")
    df = df.copy()
    df['이탈_타겟'] = df.apply(_define_churn_strict_logic, axis=1)

    # 타겟이 있는 데이터(학습용)와 없는 데이터(제외용) 분리
    train_df = df[df['이탈_타겟'].notna()].copy()
    train_df['이탈_타겟'] = train_df['이탈_타겟'].astype(int)
    
    dormant_df = df[df['이탈_타겟'].isna()].copy()

    print(f"--- 데이터 분리 결과 ---")
    print(f"1. 학습 가능 데이터: {len(train_df)}건 (이탈: {train_df['이탈_타겟'].sum()}, 유지: {len(train_df)-train_df['이탈_타겟'].sum()})")
    print(f"2. 판단 제외 데이터: {len(dormant_df)}건")
    
    return train_df, dormant_df

def remove_leakage(train_df, extra_drop_cols=None):
    """
    Data Leakage를 유발할 수 있는 컬럼들을 제거합니다.
    """
    # 1. 명시적 리키지 컬럼
    leakage_cols = [
        '당월_총_이용금액', '직전_3M_평균_이용금액',
        '이용금액_신용_B0M', '이용금액_체크_B0M',
        '이용금액_신용_R3M', '이용금액_체크_R3M'
    ]
    
    # 2. 키워드 기반 자동 제거 ('B0M', '당월' 등 현재 시점 정보)
    danger_keywords = ['B0M', '당월', '금액', '횟수', '이용', '여부', '건수', '신청']
    auto_leakage = [col for col in train_df.columns if any(word in col for word in danger_keywords)]
    
    # 3. 필수 제거 컬럼 (ID, 날짜, 타겟)
    base_drop = ['발급회원번호', '기준년월', '이탈_타겟']

    final_drop_set = set(leakage_cols + auto_leakage + base_drop)
    if extra_drop_cols:
        final_drop_set.update(extra_drop_cols)
        
    # 실제 존재하는 컬럼만 필터링
    cols_to_drop = [c for c in final_drop_set if c in train_df.columns]
    
    X = train_df.drop(columns=cols_to_drop)
    y = train_df['이탈_타겟']
    
    print(f"Leakage 제거 후 피처 수: {X.shape[1]}개 (제거된 컬럼: {len(cols_to_drop)}개)")
    
    return X, y, train_df['발급회원번호'] # ID는 Split을 위해 반환

def prepare_train_test(X, y, ids, test_size=0.2, random_state=42):
    """
    ID 기반으로 Train/Test 셋을 분리하고, 범주형 변수를 변환합니다.
    """
    unique_ids = ids.unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    # ID 기반 마스킹
    is_train = ids.isin(train_ids)
    is_test = ids.isin(test_ids)
    
    X_train = X[is_train].copy()
    y_train = y[is_train].copy()
    X_test = X[is_test].copy()
    y_test = y[is_test].copy()
    
    # 범주형 변수(object) -> category 타입 변환
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        
    print(f"Train 크기: {X_train.shape}, Test 크기: {X_test.shape}")
    print(f"범주형 피처 수: {len(cat_features)}")
    
    return X_train, y_train, X_test, y_test

def train_lgbm(X_train, y_train, X_test, y_test, params=None):
    """
    LightGBM 모델을 학습합니다.
    params가 없으면 기본 설정을 사용합니다.
    """
    if params is None:
        # 이탈 비중이 낮으므로 scale_pos_weight 등을 고려한 기본값
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'boost_from_average': False,
            'scale_pos_weight': 11, # 데이터 불균형 고려 (약 1:11 가정)
            'importance_type': 'gain'
        }
    
    print("모델 학습 시작...")
    model = LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='binary_logloss',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100)
        ]
    )
    return model

def evaluate_by_threshold(model, X_test, y_test, target_recall=None, plot_cm=False):
    """
    모델을 평가합니다. target_recall이 주어지면 해당 Recall을 달성하는 임계값을 찾아 적용합니다.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    threshold = 0.5
    if target_recall:
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        # 목표 Recall과 가장 가까운 지점 찾기
        idx = (np.abs(recalls - target_recall)).argmin()
        threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        print(f"\n[Threshold Adjustment] Target Recall: {target_recall} -> Applied Threshold: {threshold:.4f}")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    if plot_cm:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Threshold: {threshold:.4f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

def main(file_path):
    # 1. 데이터 로드
    df = load_data(file_path)
    if df is None: return

    # 2. 전처리 (기초)
    df = preprocess_initial(df)
    
    # 3. 타겟 정의
    train_df_raw, _ = define_target(df)
    
    # 4. 리키지 제거 및 X, y 분리
    X, y, ids = remove_leakage(train_df_raw)
    
    # 5. Train/Test Split
    X_train, y_train, X_test, y_test = prepare_train_test(X, y, ids)
    
    # 6. 모델 학습
    # params를 필요에 따라 수정 가능
    lgbm_params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'num_leaves': 20,
        'max_depth': 5,
        'random_state': 42,
        'scale_pos_weight': 5, # 과적합 방지를 위해 가중치 조정
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'verbose': -1
    }
    model = train_lgbm(X_train, y_train, X_test, y_test, params=lgbm_params)
    
    # 7. 평가 (예: Recall 0.86 목표)
    evaluate_by_threshold(model, X_test, y_test, target_recall=0.86)

    return model

if __name__ == "__main__":
    # 파일 경로는 실행 환경에 맞게 수정 필요
    FILE_PATH = "../../DATA/VIP/VIP/VIP_combined_part0.csv"
    
    # 직접 실행 시
    if os.path.exists(FILE_PATH):
        main(FILE_PATH)
    else:
        print(f"Warning: '{FILE_PATH}' 경로에 파일이 없습니다.")
        print("main() 함수에 올바른 파일 경로를 넣어 실행해주세요.")
