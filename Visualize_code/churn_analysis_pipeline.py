
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import linregress
import os
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# =============================================================================
# [설정] 환경 설정 및 상수 정의
# =============================================================================
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
warnings.filterwarnings('ignore')

COL_ID = '발급회원번호'
COL_DATE = '기준년월'

# 분석 변수 (Wide Format의 접미사 '_MM' 등을 뗀 순수 컬럼명 가정)
COL_SPEND = '이용금액_신용_B0M'      # 소비
COL_COUNT = '이용건수_신용_B0M'      # 빈도
COL_COUNT_SPEND = '이용건수_신판_B0M' # 빈도 소비 (할부, 일시불)
COL_BALANCE = '잔액_B0M'             # 잔액
COL_CASH_ADV = '잔액_현금서비스_B0M' # 악성부채1
COL_CARD_LOAN = '잔액_카드론_B0M'    # 악성부채2
COL_DELINQ = '연체잔액_B0M'          # 리스크3
COL_AVG_BAL = '월중평잔'             # 자산

# Additional columns for R12M fallback
COL_SPEND_R12M = '이용금액_신용_R12M'
COL_COUNT_R12M = '이용건수_신용_R12M'

# 데이터 누수 방지를 위한 컬럼 리스트
LEAKAGE_COLS = [
    'Target', '발급회원번호', 'Unnamed: 0', '기준년월',
    'Slope_Spend', 'Slope_Balance', 'Slope_Count', 
    'Norm_Slope_Spend', 'Norm_Slope_Balance', 'Norm_Slope_Count',
    'Score_BadDebt', 'Score_Delinq', 'Score_Activity', 'Score_Asset',
    'Score_Status_Total', 'Score_Slope_Total', 'Final_Total_Score',
    'Risk_Count', 'Churn_Segment',
    'Cond1_Has_BadDebt', 'Cond2_Has_Delinq', 'Cond3_Activity_Drop', 'Cond4_Asset_Zero',
    '이용건수_신용_R6M', '이용건수_신용_R3M', '이용건수_일시불_R6M', '이용건수_신판_R3M', '이용건수_신판_R6M', '이용건수_신용_B0M',
    '이용건수_일시불_R3M','최종이용일자_기본', '이용건수_신판_B0M','최종이용일자_신판', '이용건수_일시불_B0M',
    '최종이용일자_일시불', '이용후경과월_일시불', '_1순위카드이용건수'
]

# =============================================================================
# [Helper Functions] 계산 및 로직
# =============================================================================
def calc_slope_long(series):
    """시계열 데이터(Series)의 선형 회귀 기울기를 계산"""
    y = series.values.astype(float)
    if len(y) < 2 or np.sum(y) == 0:
        return 0
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    return 0 if np.isnan(slope) else slope

def calculate_churn_scores(group):
    """고객 한 명의 데이터를 받아 점수 및 Target 생성 (1개월 이상 데이터 필요)"""
    # 데이터가 아예 없는 경우
    if len(group) < 1:
        return pd.Series({
            'Score_BadDebt': 0, 'Score_Delinq': 0, 'Score_Activity': 0, 'Score_Asset': 0,
            'Score_Status_Total': 0, 'Slope_Spend': 0, 'Slope_Balance': 0, 'Slope_Count': 0
        })

    # (A) 상태 점수 (Status Score) 세부 항목 계산
    try:
        # Helper for safe indexing
        def get_val(col, idx_from_last):
            if len(group) >= idx_from_last:
                return group[col].iloc[-idx_from_last]
            return 0

        # 1. [부정] 악성 부채 점수 (Score_BadDebt)
        val_last = get_val(COL_CASH_ADV, 1)
        val_prev = get_val(COL_CASH_ADV, 2)
        
        loan_last = get_val(COL_CARD_LOAN, 1)
        loan_prev = get_val(COL_CARD_LOAN, 2)
        
        bad_debt_score = (
            ((val_last - val_prev) / (val_prev + 1) * 1.5) +
            ((loan_last - loan_prev) / (loan_prev + 1) * 1.0)
        )
        
        # 2. [부정] 연체 강도 점수 (Score_Delinq)
        delinq_score = (get_val(COL_DELINQ, 1) * 3.0) + (get_val(COL_DELINQ, 2) * 2.0)
        if len(group) >= 3:
            delinq_score += (get_val(COL_DELINQ, 3) * 1.0)
        
        # 3. [긍정] 활동성 점수 (Score_Activity)
        sum_r3 = group[COL_COUNT_SPEND].iloc[-3:].sum()
        sum_r6 = group[COL_COUNT_SPEND].sum()
        activity_score = ((sum_r3 * 2) - sum_r6) / (sum_r6 + 1) * 100
        
        # 4. [긍정] 자산 방어 점수 (Score_Asset)
        avg_r3 = group[COL_AVG_BAL].iloc[-3:].mean()
        avg_r6 = group[COL_AVG_BAL].mean()
        asset_score = (avg_r3 / (avg_r6 + 1)) * 10
        
        # >> [Total] 상태 종합 점수 (Score_Status_Total)
        score_status_total = (bad_debt_score + delinq_score) - (activity_score + asset_score)
    except:
        bad_debt_score = 0
        delinq_score = 0
        activity_score = 0
        asset_score = 0
        score_status_total = 0

    # (B) 기울기 점수 (Slope Score)
    # CASE 1: Data >= 2 months (Use linregress)
    if len(group) >= 2:
        slope_spend = calc_slope_long(group[COL_SPEND])
        slope_balance = calc_slope_long(group[COL_BALANCE])
        slope_count = calc_slope_long(group[COL_COUNT])
    
    # CASE 2: Data == 1 month (Use R12M fallback)
    else:
        # Spending Slope Proxy: Current - Monthly_Avg(R12M)
        r12m_spend = group[COL_SPEND_R12M].iloc[0] if COL_SPEND_R12M in group.columns else 0
        avg_spend = r12m_spend / 12
        slope_spend = group[COL_SPEND].iloc[0] - avg_spend
        
        # Count Slope Proxy
        r12m_count = group[COL_COUNT_R12M].iloc[0] if COL_COUNT_R12M in group.columns else 0
        avg_count = r12m_count / 12
        slope_count = group[COL_COUNT].iloc[0] - avg_count
        
        # Balance Slope: Set to -1 (safe condition) as requested
        slope_balance = -1

    return pd.Series({
        'Score_BadDebt': bad_debt_score,
        'Score_Delinq': delinq_score,
        'Score_Activity': activity_score,
        'Score_Asset': asset_score,
        'Score_Status_Total': score_status_total,
        'Slope_Spend': slope_spend,
        'Slope_Balance': slope_balance,
        'Slope_Count': slope_count
    })

def check_churn_condition(scores):
    """Calculates Target (1 or 0) from scores series"""
    # (조건 A) 기울기 3종(소비, 잔액, 건수)이 모두 0 이하
    cond_slopes_decrease = (
        (scores['Slope_Spend'] <= 0) & 
        (scores['Slope_Balance'] <= 0) & 
        (scores['Slope_Count'] <= 0)
    )
    
    # (조건 B) 4대 위험 징후 중 1개 이상 감지 (Risk_Count >= 1 로 수정됨)
    cond1 = scores['Score_BadDebt'] > 0
    cond2 = scores['Score_Delinq'] > 0
    cond3 = scores['Score_Activity'] < 0
    cond4 = scores['Score_Asset'] == 0
    
    risk_count = int(cond1) + int(cond2) + int(cond3) + int(cond4)
    cond_high_risk = (risk_count >= 1)
    
    return 1 if (cond_slopes_decrease and cond_high_risk) else 0

# =============================================================================
# [Analysis Functions] 데이터 로드 및 분석
# =============================================================================
def analyze_rolling_churn(file_path):
    """이탈자 Rolling 분석: 이탈 지속 기간 확인"""
    print(f"\n[Info] 파일 로드 및 분석 시작: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None, None

    try:
        df = pd.read_csv(file_path, low_memory=False)
        required_cols = [
            COL_ID, COL_DATE, COL_SPEND, COL_COUNT, COL_BALANCE, 
            COL_CASH_ADV, COL_CARD_LOAN, COL_DELINQ, COL_AVG_BAL,
            COL_SPEND_R12M, COL_COUNT_R12M, COL_COUNT_SPEND
        ]
        # 없는 컬럼 0으로 채우기
        for c in required_cols:
            if c not in df.columns: df[c] = 0
            
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None, None

    # Sort
    df.sort_values(by=[COL_ID, COL_DATE], inplace=True)
    grouped = df.groupby(COL_ID)
    
    results = []
    print(" - 고객별 Rolling Analysis 진행 중...")
    
    count_churners = 0
    
    for cust_id, group in grouped:
        if len(group) < 1: continue
            
        current_scores = calculate_churn_scores(group)
        is_current_churn = check_churn_condition(current_scores)
        
        if is_current_churn == 1:
            count_churners += 1
            consecutive_months = 1 
            max_lookback = len(group) - 1
            
            for i in range(1, max_lookback + 1):
                past_group = group.iloc[:-i] 
                past_scores = calculate_churn_scores(past_group)
                is_past_churn = check_churn_condition(past_scores)
                
                if is_past_churn == 1:
                    consecutive_months += 1
                else:
                    break 
            
            results.append({
                COL_ID: cust_id,
                'Churn_Duration_Months': consecutive_months
            })

    if not results:
        print("❌ 분석된 이탈자가 없습니다.")
        return None, df

    df_res = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print(f"📊 이탈자 Rolling 분석 결과")
    print("="*50)
    print(f" - 총 분석 고객 수: {len(grouped)}명")
    print(f" - 최종 시점 이탈자 수: {len(df_res)}명 ({len(df_res)/len(grouped)*100:.2f}%)")
    print("-" * 30)
    print(" [이탈 징후 지속 기간 통계]")
    print(df_res['Churn_Duration_Months'].describe())
    
    return df_res, df

def analyze_and_extract_features_v2(input_data):
    """특성 추출 및 ML용 데이터 준비 (점수 데이터 생성)"""
    if isinstance(input_data, pd.DataFrame):
        print(f"\n[Info] DataFrame 입력됨 - 분석 시작")
        df = input_data.copy()
    elif isinstance(input_data, str) and os.path.exists(input_data):
        print(f"\n[Info] 파일 로드 및 분석 시작: {input_data}")
        df = pd.read_csv(input_data, low_memory=False)
    else:
        print("❌ 유효한 데이터가 아닙니다.")
        return None, None

    # 필수 컬럼 체크
    required_cols = [
        COL_ID, COL_DATE, COL_SPEND, COL_COUNT, COL_BALANCE, 
        COL_CASH_ADV, COL_CARD_LOAN, COL_DELINQ, COL_AVG_BAL, 
        COL_COUNT_SPEND
    ]
    for c in required_cols:
        if c not in df.columns: df[c] = 0

    df.sort_values(by=[COL_ID, COL_DATE], inplace=True)
    grouped = df.groupby(COL_ID)
    
    results = []
    print(" - 고객별 Feature Extraction 진행 중 (Scores 계산)...")
    
    for cust_id, group in grouped:
        if len(group) < 1: continue
        
        scores = calculate_churn_scores(group)
        is_churn = check_churn_condition(scores)
        
        row_data = scores.to_dict()
        row_data[COL_ID] = cust_id
        row_data['Target'] = is_churn
        results.append(row_data)
        
    df_res = pd.DataFrame(results)
    print(f"✅ Feature Extraction 완료: {len(df_res)}건")
    return df_res, df

def make_ml_dataset_final(df_raw, df_scores_viz=None):
    """ML용 최종 데이터셋 병합 (Features + Target)"""
    print(f"[Info] 모델 학습용 최종 데이터셋 생성 시작")

    # 1. Target 데이터 준비
    if df_scores_viz is not None:
        print(" - 기존 Score 데이터 활용")
        df_target = df_scores_viz.copy()
    else:
        print(" - Score 데이터 새로 계산")
        df_target = df_raw.groupby(COL_ID).apply(calculate_churn_scores).reset_index()

    # Target 라벨링 (중복 계산 방지 위해 로직 재적용)
    cond_slopes_decrease = (
        (df_target['Slope_Spend'] <= 0) & 
        (df_target['Slope_Balance'] <= 0) & 
        (df_target['Slope_Count'] <= 0)
    )
    
    score_cols = ['Score_BadDebt', 'Score_Delinq', 'Score_Activity', 'Score_Asset']
    risk_count = 0
    for col in score_cols:
        if col in df_target.columns:
            if col == 'Score_Activity':
                risk_count += (df_target[col] < 0).astype(int)
            elif col == 'Score_Asset':
                risk_count += (df_target[col] == 0).astype(int)
            else:
                risk_count += (df_target[col] > 0).astype(int)
                
    cond_high_risk = (risk_count >= 1)
    df_target['Target'] = np.where(cond_slopes_decrease & cond_high_risk, 1, 0)

    # 2. Raw Features (최신 데이터) 추출
    print(" - 고객별 최신 데이터 추출 중...")
    df_features = df_raw.sort_values(by=[COL_ID, COL_DATE]).groupby(COL_ID).last().reset_index()

    # 3. 병합
    print(" - 데이터 병합 중...")
    if COL_ID not in df_target.columns and df_target.index.name == COL_ID:
        df_target = df_target.reset_index()
        
    df_final = pd.merge(df_features, df_target[[COL_ID, 'Target']], on=COL_ID, how='inner')
    
    print(f"✅ 최종 데이터셋 생성 완료: {len(df_final)}명")
    print(f" - Target 분포:\n{df_final['Target'].value_counts()}")
    
    return df_final

# =============================================================================
# [Model Training Functions] 모델 학습
# =============================================================================
def run_rf_simulation(data, drop_cols=LEAKAGE_COLS):
    print(f"[Info] Random Forest 학습 시작")
    
    # 전처리
    data_clean = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    targets_to_drop = [COL_ID, 'Target'] + drop_cols
    X_temp = data_clean.drop(columns=targets_to_drop, errors='ignore')
    y = data_clean['Target']
    X = X_temp.select_dtypes(include=['number'])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 학습
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # 평가
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"📊 RF 결과: Accuracy={acc:.4f}, ROC-AUC={roc:.4f}")
    print(classification_report(y_test, y_pred))
    
    return rf, acc, roc

def run_xgboost_simulation(data, drop_cols=LEAKAGE_COLS):
    print(f"[Info] XGBoost 학습 시작")
    
    data_clean = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    targets_to_drop = [COL_ID, 'Target'] + drop_cols
    X_temp = data_clean.drop(columns=targets_to_drop, errors='ignore')
    y = data_clean['Target']
    X = X_temp.select_dtypes(include=['number'])
    
    # 컬럼명 특수문자 제거
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = ["".join(x.split()) for x in X.columns]
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_weight = count_neg / count_pos if count_pos > 0 else 1

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=8,
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1, tree_method='hist'
    )
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"📊 XGB 결과: Accuracy={acc:.4f}, ROC-AUC={roc:.4f}")
    
    return xgb_model, acc, roc

def run_lightgbm_simulation(data, drop_cols=LEAKAGE_COLS):
    print(f"[Info] LightGBM 학습 시작")
    
    data_clean = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    targets_to_drop = [COL_ID, 'Target'] + drop_cols
    X_temp = data_clean.drop(columns=targets_to_drop, errors='ignore')
    y = data_clean['Target']
    X = X_temp.select_dtypes(include=['number'])
    
    regex = re.compile(r"[\[\]<>\s,]", re.IGNORECASE)
    X.columns = [regex.sub("_", str(col)) for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_weight = count_neg / count_pos if count_pos > 0 else 1

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=31,
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    
    y_pred = lgbm_model.predict(X_test)
    y_prob = lgbm_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"📊 LGBM 결과: Accuracy={acc:.4f}, ROC-AUC={roc:.4f}")
    
    return lgbm_model, acc, roc

# =============================================================================
# [Visualization] 시각화
# =============================================================================
def compare_existing_models(models_dict, data, drop_cols=LEAKAGE_COLS):
    """학습된 모델들 비교 시각화 (데이터셋을 다시 Split하여 동일 조건 평가)"""
    print(f"\n[Info] 모델 비교 분석 시작...")
    
    # Test set 준비 (동일한 Random State 사용)
    data_clean = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    targets_to_drop = [COL_ID, 'Target'] + drop_cols
    X_temp = data_clean.drop(columns=targets_to_drop, errors='ignore')
    y = data_clean['Target']
    X = X_temp.select_dtypes(include=['number'])
    
    # 컬럼명 전처리 (LGBM/XGB 호환)
    regex = re.compile(r"[\[\]<>\s,]", re.IGNORECASE)
    X.columns = [regex.sub("_", str(col)) for col in X.columns]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    plt.figure(figsize=(10, 6))
    for name, model in models_dict.items():
        if model is None: continue
        
        # 모델마다 feature 이름이 다를 수 있어 try-except 처리 또는 재학습 권장되나
        # 여기서는 같은 순서 전제로 진행
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc:.4f})")
        except Exception as e:
            print(f"⚠️ {name} 예측 실패: {e}")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC Curve Comparison')
    plt.legend()
    plt.show()

# =============================================================================
# [Main] 메인 실행부
# =============================================================================
if __name__ == "__main__":
    # 데이터 경로 설정 (필요시 수정)
    TARGET_FILE_PATH = "260108/general_combined_part0.csv" # 예시 파일명
    
    if os.path.exists(TARGET_FILE_PATH):
        # 1. 데이터 로드 및 점수 생성
        df_viz, df_raw = analyze_and_extract_features_v2(TARGET_FILE_PATH)
        
        if df_viz is not None:
            # 2. 이탈 분석 (옵션)
            # analyze_rolling_churn(TARGET_FILE_PATH)
            
            # 3. 모델 학습용 데이터셋 병합
            df_ml = make_ml_dataset_final(df_raw, df_viz)
            
            # 4. 모델 학습 실행
            rf_model, rf_acc, rf_roc = run_rf_simulation(df_ml)
            xgb_model, xgb_acc, xgb_roc = run_xgboost_simulation(df_ml)
            lgbm_model, lgb_acc, lgb_roc = run_lightgbm_simulation(df_ml)
            
            # 5. 비교 시각화
            models = {
                'Random Forest': rf_model,
                'XGBoost': xgb_model,
                'LightGBM': lgbm_model
            }
            compare_existing_models(models, df_ml)
            
            print("\n✅ 모든 프로세스가 완료되었습니다.")
        else:
            print("데이터 분석 실패")
    else:
        print(f"파일을 찾을 수 없습니다: {TARGET_FILE_PATH}")
        print("스크립트 하단의 TARGET_FILE_PATH 변수를 실제 데이터 경로로 수정하여 실행하세요.")
