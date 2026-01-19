import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

# 경고 무시
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    CSV 데이터를 불러옵니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. ({file_path})")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료: {df.shape}")
        return df
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return None

def preprocess_and_define_target(df):
    """
    전처리 및 타겟 정의 (Method B strict)
    """
    df = df.sort_values(by=['발급회원번호', '기준년월']).copy()

    # 파생변수 생성
    if '이용금액_신용_B0M' in df.columns and '이용금액_체크_B0M' in df.columns:
        df['당월_총_이용금액'] = df['이용금액_신용_B0M'] + df['이용금액_체크_B0M']
    
    if '이용금액_신용_R3M' in df.columns and '이용금액_체크_R3M' in df.columns:
        df['직전_3M_평균_이용금액'] = (df['이용금액_신용_R3M'] + df['이용금액_체크_R3M']) / 3

    # 타겟 정의 로직 (Method B strict)
    def define_churn(row):
        if pd.isna(row.get('직전_3M_평균_이용금액')) or row.get('직전_3M_평균_이용금액') <= 0:
            return np.nan
        if row.get('당월_총_이용금액') < (row.get('직전_3M_평균_이용금액') * 0.8):
            return 1
        return 0

    df['이탈_타겟'] = df.apply(define_churn, axis=1)

    # 학습용 데이터만 추출
    train_df = df[df['이탈_타겟'].notna()].copy()
    train_df['이탈_타겟'] = train_df['이탈_타겟'].astype(int)
    
    print(f"학습 가능 데이터: {len(train_df)}건 (이탈: {train_df['이탈_타겟'].sum()})")
    
    return train_df

def remove_leakage_and_split(train_df, test_size=0.2, random_state=42):
    """
    Leakage 컬럼 제거 및 ID 기반 Train/Test Split
    """
    target_col = '이탈_타겟'
    
    # 제거할 컬럼 리스트
    leakage_cols = [
        '당월_총_이용금액', '직전_3M_평균_이용금액',
        '이용금액_신용_B0M', '이용금액_체크_B0M', '이용금액_신용_R3M', '이용금액_체크_R3M'
    ]
    b0m_cols = [col for col in train_df.columns if 'B0M' in col]
    base_drop = ['발급회원번호', '기준년월', '이탈_타겟']

    final_drop_set = set(leakage_cols + b0m_cols + base_drop)
    cols_to_drop = [c for c in final_drop_set if c in train_df.columns]
    
    X = train_df.drop(columns=cols_to_drop)
    y = train_df[target_col]
    
    # ID 기반 Split
    unique_ids = train_df['발급회원번호'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    X_train = X[train_df['발급회원번호'].isin(train_ids)].copy()
    y_train = y[train_df['발급회원번호'].isin(train_ids)].copy()
    
    X_test = X[train_df['발급회원번호'].isin(test_ids)].copy()
    y_test = y[train_df['발급회원번호'].isin(test_ids)].copy()
    
    print(f"Train 크기: {X_train.shape}, Test 크기: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def _xgb_supports_native_categorical():
    """XGBoost 버전에 따라 native categorical 지원 여부 확인"""
    try:
        major, minor = xgb.__version__.split('.')[:2]
        major, minor = int(major), int(minor)
        return (major > 1) or (major == 1 and minor >= 6)
    except Exception:
        return False

def prepare_for_xgboost(X_train, X_test):
    """
    XGBoost용 전처리: 
    Native Categorical 지원 시 -> category 타입 변환
    미지원 시 -> One-Hot Encoding
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    cat_cols = X_tr.select_dtypes(include=['object']).columns.tolist()
    
    if not cat_cols:
        return X_tr, X_te, cat_cols, True
        
    # 우선 category로 변환
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype('category')
        X_te[col] = X_te[col].astype('category')
        
    use_native = _xgb_supports_native_categorical()
    
    if use_native:
        print(f"XGBoost Native Categorical 사용 (범주형 변수: {len(cat_cols)}개)")
        return X_tr, X_te, cat_cols, True
    else:
        print(f"One-Hot Encoding 적용 (범주형 변수: {len(cat_cols)}개)")
        # One-Hot Encoding Fallback
        X_tr_enc = pd.get_dummies(X_tr, columns=cat_cols, dummy_na=True)
        X_te_enc = pd.get_dummies(X_te, columns=cat_cols, dummy_na=True)
        # Train/Test 컬럼 정렬 (Align)
        X_tr_enc, X_te_enc = X_tr_enc.align(X_te_enc, join='outer', axis=1, fill_value=0)
        return X_tr_enc, X_te_enc, cat_cols, False

def train_xgboost(X_train, y_train, X_test, y_test, use_native_cat):
    """
    XGBoost 모델 학습
    """
    # 불균형 가중치 계산
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    
    model = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.0,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=use_native_cat
    )
    
    print("모델 학습 시작 (XGBoost)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=200,
        early_stopping_rounds=100
    )
    
    return model

def evaluate_and_save(model, X_test, y_test, target_recall, output_file):
    """
    모델 평가 및 결과 저장
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 기본 평가 (Threshold 0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\n--- Threshold = 0.5 ---")
    print(classification_report(y_test, y_pred_default))
    
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    except:
        pass
        
    # 목표 Recall에 맞춘 Threshold 튜닝
    if target_recall:
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        # recalls는 thresholds보다 길이가 1 김
        idx = (np.abs(recalls[:-1] - target_recall)).argmin()
        best_thr = float(thresholds[idx])
        
        y_pred_thr = (y_proba >= best_thr).astype(int)
        
        print(f"\n✅ 목표 Recall {target_recall} 근처 Threshold = {best_thr:.6f}")
        print(classification_report(y_test, y_pred_thr))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_thr)
        print("Confusion Matrix (tuned):\n", cm)
        
    # 모델 저장
    model.save_model(output_file)
    print(f"\n모델 저장 완료: {output_file}")
    
    return y_proba

def main(file_path, output_model_path="VIP_XGBoost_model.json"):
    # 1. 데이터 로드
    df = load_data(file_path)
    if df is None: return

    # 2. 전처리 및 타겟 정의
    train_df = preprocess_and_define_target(df)
    
    # 3. Leakage 제거 및 Split
    X_train, y_train, X_test, y_test = remove_leakage_and_split(train_df)
    
    # 4. XGBoost용 전처리 (범주형 변수)
    X_train_p, X_test_p, _, use_native_cat = prepare_for_xgboost(X_train, X_test)
    
    # 5. 학습
    model = train_xgboost(X_train_p, y_train, X_test_p, y_test, use_native_cat)
    
    # 6. 평가 및 저장 (목표 Recall 0.86)
    evaluate_and_save(model, X_test_p, y_test, target_recall=0.86, output_file=output_model_path)

if __name__ == "__main__":
    # 실행 예시
    FILE_PATH = "../../DATA/VIP/VIP/VIP_combined_part0.csv"
    
    if os.path.exists(FILE_PATH):
        main(FILE_PATH)
    else:
        print(f"Warning: '{FILE_PATH}' 파일이 없습니다.")
        print("스크립트 하단의 FILE_PATH를 수정하거나 main() 함수를 직접 호출하세요.")
