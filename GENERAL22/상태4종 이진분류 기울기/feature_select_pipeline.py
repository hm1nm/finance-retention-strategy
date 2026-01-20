import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import shap
from scipy.stats import linregress
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  # VIF 상수항 추가용
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import os
import platform # OS 확인용

# ========================================================================================
# 1. Configuration & Constants
# ========================================================================================
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Windows
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False

COL_ID = '발급회원번호'
COL_DATE = '기준년월'

# [수정됨] 실제 데이터 명세서(필드한글명) 기반 매핑
# 3. 승인매출 정보.csv 참고
COL_SPEND = '이용금액_신용_B0M'       # 기존: 당월_총_이용금액 -> 변경: 이용금액_신용_B0M (일시불+할부+현금서비스+카드론)
COL_COUNT = '이용건수_신용_B0M'       # 기존: 당월_총_이용건수 -> 변경: 이용건수_신용_B0M

# 5. 잔액 정보.csv 참고
COL_BALANCE = '잔액_B0M'             # 기존: 당월_신용공여_총_잔액 -> 변경: 잔액_B0M
COL_AVG_BAL = '평잔_3M'              # 기존: 최근_3개월_평균_잔액 -> 변경: 평잔_3M

# 3. 승인매출 정보.csv 참고
COL_CASH_ADV = '이용금액_CA_B0M'      # 기존: 당월_현금서비스_이용금액 -> 변경: 이용금액_CA_B0M
COL_CARD_LOAN = '이용금액_카드론_B0M'  # 기존: 당월_카드론_이용금액 -> 변경: 이용금액_카드론_B0M

# 1. 회원 정보.csv 참고
COL_DELINQ = '회원여부_연체'          # 기존: 당월_연체_여부 -> 변경: 회원여부_연체 (0:미연체, 1:연체)

# Rolling 12 Months (R12M) Columns (1년치 데이터가 있을 경우 사용)
# 명세서에 해당 컬럼들이 존재하므로 매핑합니다.
COL_SPEND_R12M = '이용금액_신용_R12M' # 최근 1년간 이용금액
COL_COUNT_R12M = '이용건수_신용_R12M' # 최근 1년간 이용건수


# ========================================================================================
# 2. Utility Functions
# ========================================================================================
def calc_slope_long(series):
    """
    Calculate the slope of a linear regression line for a given series.
    Returns 0 if the series has fewer than 2 data points or variance is zero.
    """
    y = series.values
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    if np.all(y == y[0]):
        return 0.0
    slope, _, _, _, _ = linregress(x, y)
    if np.isnan(slope):
        return 0.0
    return slope

def normalize_risk_vector(series):
    """
    Normalize slope values to a risk score (0-1).
    Negative slope (decreasing trend) -> Higher Risk (closer to 1).
    Positive slope (increasing trend) -> Lower Risk (closer to 0).
    """
    if series.empty:
        return series
    
    # We want decreasing trend (negative slope) to be high risk.
    # So we inverse the values: risk_raw = -slope
    risk_raw = -series
    
    # Min-Max Scaling to 0~1
    min_val = risk_raw.min()
    max_val = risk_raw.max()
    
    if max_val == min_val:
        return pd.Series(0, index=series.index)
        
    normalized = (risk_raw - min_val) / (max_val - min_val)
    return normalized

def calculate_vif(dataframe, sample_size=5000):
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.
    Samples data if it's too large. Adds constant for correct calculation.
    """
    print("\n🔍 [VIF Check] Calculating Variance Inflation Factors...")
    
    df_vif_input = dataframe.select_dtypes(include=[np.number]).dropna()
    
    # Sampling if data is large
    if len(df_vif_input) > sample_size:
        print(f" - Data size ({len(df_vif_input)}) is large. Sampling {sample_size} rows for VIF calculation.")
        df_vif_input = df_vif_input.sample(n=sample_size, random_state=42)
        
    # Remove leakage/target columns if present
    cols_to_exclude = ['Target', '발급회원번호', 'Unnamed: 0', 'index']
    cols_check = [c for c in df_vif_input.columns if c not in cols_to_exclude]
    df_vif_input = df_vif_input[cols_check]
    
    # [수정] VIF 계산 전 상수항 추가 (필수)
    df_vif_input = add_constant(df_vif_input)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_vif_input.columns
    
    try:
        vif_data["VIF"] = [variance_inflation_factor(df_vif_input.values, i) 
                            for i in range(df_vif_input.shape[1])]
    except Exception as e:
        print(f"⚠️ VIF calculation failed: {e}")
        return None

    # 상수항(const) 행 제거 후 정렬
    vif_data = vif_data[vif_data['Feature'] != 'const']
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    print(vif_data)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x="VIF", y="Feature", data=vif_data.head(20))
    plt.title("Top 20 Features by VIF")
    plt.xlabel("VIF Score")
    plt.tight_layout()
    plt.show()
    
    return vif_data


# ========================================================================================
# 3. Core Logic: Scoring & Target Generation
# ========================================================================================

def calculate_churn_scores(group):
    """
    Calculate Churn Scores for a customer group (sorted by date).
    Handles R12M fallback for customers with only 1 month of data.
    """
    months_data = len(group)
    
    # --- Slope Calculation Logic ---
    if months_data >= 2:
        # Normal Slope Calculation
        slope_spend = calc_slope_long(group[COL_SPEND])
        slope_balance = calc_slope_long(group[COL_BALANCE])
        slope_count = calc_slope_long(group[COL_COUNT])
    else:
        # Fallback for 1-month data using R12M
        current_spend = group[COL_SPEND].iloc[-1]
        r12m_spend = group[COL_SPEND_R12M].iloc[-1] if COL_SPEND_R12M in group.columns else 0
        slope_spend = current_spend - r12m_spend
        
        current_count = group[COL_COUNT].iloc[-1]
        r12m_count = group[COL_COUNT_R12M].iloc[-1] if COL_COUNT_R12M in group.columns else 0
        slope_count = current_count - r12m_count
        
        # Balance Slope Fallback -> Set to -1 (Risk) manually as per analysis
        slope_balance = -1 
        
    score_status_total = 0
    
    # 예외 처리: 컬럼이 없을 경우를 대비해 get으로 안전하게 가져오기
    delinq_sum = group[COL_DELINQ].sum() if COL_DELINQ in group.columns else 0
    cash_adv_sum = group[COL_CASH_ADV].sum() if COL_CASH_ADV in group.columns else 0

    if delinq_sum > 0:
        score_status_total += 50
        
    if cash_adv_sum > 0:
        score_status_total += 30
    
    return pd.Series({
        'Score_Status_Total': score_status_total,
        'Slope_Spend': slope_spend,
        'Slope_Balance': slope_balance,
        'Slope_Count': slope_count,
        'Score_BadDebt': 1 if delinq_sum > 0 else 0, 
        'Score_Delinq': 1 if delinq_sum > 0 else 0,
        'Score_Activity': -1 if slope_count < 0 else 0,
        'Score_Asset': 0 
    })

def check_churn_condition(df_scores):
    """
    Generate 'Target' based on scores.
    """
    print(" - Generating Target Variable...")
    
    # 1. Slope Condition (Decreasing Trend)
    cond_slopes_decrease = (
        (df_scores['Slope_Spend'] <= 0) & 
        (df_scores['Slope_Balance'] <= 0) & 
        (df_scores['Slope_Count'] <= 0)
    )
    
    # 2. Risk Count Condition
    cond1 = df_scores['Score_BadDebt'] > 0
    cond2 = df_scores['Score_Delinq'] > 0
    cond3 = df_scores['Score_Activity'] < 0
    cond4 = df_scores['Score_Asset'] == 0 
    
    risk_count = cond1.astype(int) + cond2.astype(int) + cond3.astype(int) + cond4.astype(int)
    cond_high_risk = (risk_count >= 1)
    
    # Final Target
    df_scores['Target'] = np.where(cond_slopes_decrease & cond_high_risk, 1, 0)
    
    # Calculate Total Scores for Analysis
    norm_slope_spend = normalize_risk_vector(df_scores['Slope_Spend']) * 30
    norm_slope_count = normalize_risk_vector(df_scores['Slope_Count']) * 30
    norm_slope_bal = normalize_risk_vector(df_scores['Slope_Balance']) * 40 
    
    df_scores['Score_Slope_Total'] = norm_slope_spend + norm_slope_count + norm_slope_bal
    df_scores['Final_Total_Score'] = (df_scores['Score_Status_Total'] + df_scores['Score_Slope_Total']) * 0.5
    
    return df_scores

def process_data_and_merge(file_path):
    print(f"\n1. [Data Load] Loading {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        # 한글 경로 에러 방지용 engine='python'
        df = pd.read_csv(file_path, low_memory=False) # engine='python'
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None
        
    if COL_SPEND_R12M not in df.columns: df[COL_SPEND_R12M] = 0
    if COL_COUNT_R12M not in df.columns: df[COL_COUNT_R12M] = 0

    print(" - Sorting data...")
    df.sort_values(by=[COL_ID, COL_DATE], inplace=True)
    
    print("2. [Scoring] Calculating Churn Scores...")
    # pandas 최신 버전 대응 (include_groups=False 권장되나 호환성을 위해 유지하되 에러시 수정)
    try:
        df_scores = df.groupby(COL_ID).apply(calculate_churn_scores).reset_index()
    except Exception as e:
        print(f"Grouping Error: {e}")
        return None
        
    df_scores = check_churn_condition(df_scores)
    
    print(f" - Target Ratio: {df_scores['Target'].value_counts(normalize=True).to_dict()}")
    
    print("3. [Merge] Merging Scores with Features...")
    df_last = df.groupby(COL_ID).tail(1).copy()
    
    df_final = df_last.merge(df_scores, on=COL_ID, how='left')
    
    return df_final


# ========================================================================================
# 4. Visualization Modules
# ========================================================================================

def plot_score_distributions(df, target_col='Target'):
    print("\n📊 [Distribution Analysis] Plotting Score Distributions...")
    
    cols_to_plot = ['Final_Total_Score', 'Score_Slope_Total', 'Score_Status_Total', 
                    'Slope_Spend', 'Slope_Count', 'Slope_Balance']
    
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]
    
    if not cols_to_plot:
        print("⚠️ No score columns found to plot.")
        return

    fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=2, figsize=(15, 4 * len(cols_to_plot)))
    
    for i, col in enumerate(cols_to_plot):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, element="step", ax=axes[i, 0], palette='Set1')
        axes[i, 0].set_title(f'{col} Distribution by Target')
        
        sns.boxplot(data=df, x=target_col, y=col, ax=axes[i, 1], palette='Set1')
        axes[i, 1].set_title(f'{col} Boxplot by Target')
        
    plt.tight_layout()
    plt.show()
    print("✅ Distribution plots displayed.")

def plot_confusion_matrix_heatmap(y_test, y_pred, title):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_importance(model, title):
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=20, height=0.5, title=f'Feature Importance - {title}')
    plt.tight_layout()
    plt.show()
    
def visualize_shap_summary(model, X_train, top_n_shap=20):
    print("\n📊 [SHAP Analysis] Calculating SHAP values...")
    
    X_shap = X_train
    if len(X_train) > 2000:
        X_shap = X_train.sample(n=2000, random_state=42)
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # 이진 분류일 경우 리스트로 반환될 수 있음 (class 0, class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    plt.figure()
    shap.summary_plot(shap_values, X_shap, max_display=top_n_shap)
    plt.show()


# ========================================================================================
# 5. Model Training Modules
# ========================================================================================

def train_eval_lgbm(df_train, target_col='Target'):
    print(f"\n🏋️ [Model Training] LightGBM...")
    
    leakage_cols = [
        target_col, COL_ID, 'Unnamed: 0',
        'Slope_Spend', 'Slope_Balance', 'Slope_Count',
        'Score_BadDebt', 'Score_Delinq', 'Score_Activity', 'Score_Asset',
        'Score_Status_Total', 'Score_Slope_Total', 'Final_Total_Score'
    ]
    
    features = [c for c in df_train.columns if c not in leakage_cols]
    numeric_features = df_train[features].select_dtypes(include=[np.number]).columns
    
    X = df_train[numeric_features]
    y = df_train[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1 # 경고 메시지 억제
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n[LightGBM Results]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))
    
    plot_confusion_matrix_heatmap(y_test, y_pred, "LightGBM")
    plot_feature_importance(model, "LightGBM")
    
    return model, X_train

def train_eval_xgboost_shap(df_train, target_col='Target'):
    print(f"\n🏋️ [Model Training] XGBoost for SHAP...")
    
    leakage_cols = [
        target_col, COL_ID, 'Unnamed: 0',
        'Slope_Spend', 'Slope_Balance', 'Slope_Count',
        'Score_BadDebt', 'Score_Delinq', 'Score_Activity', 'Score_Asset',
        'Score_Status_Total', 'Score_Slope_Total', 'Final_Total_Score'
    ]
    
    features = [c for c in df_train.columns if c not in leakage_cols]
    numeric_features = df_train[features].select_dtypes(include=[np.number]).columns
    
    X = df_train[numeric_features]
    y = df_train[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    visualize_shap_summary(model, X_train)
    
    return model


# ========================================================================================
# 6. Main Execution
# ========================================================================================

# File Paths
DATA_FILE_PATH = './260108/general_combined_part1.csv'

if __name__ == "__main__":
    print("🚀 Starting Feature Selection & Visualization Pipeline...")
    
    # ⚠️ 중요: DATA_FILE_PATH가 실제 파일 위치와 일치하는지 확인해주세요.
    # 업로드 하신 파일들을 먼저 병합(merge)하여 하나의 csv로 만드는 작업이 선행되었는지 확인이 필요합니다.
    if os.path.exists(DATA_FILE_PATH):
        df_final = process_data_and_merge(DATA_FILE_PATH)
        
        if df_final is not None:
            plot_score_distributions(df_final)
            calculate_vif(df_final)
            lgbm_model, X_train_lgbm = train_eval_lgbm(df_final)
            xgb_model = train_eval_xgboost_shap(df_final)
            print("✅ Pipeline Completed Successfully.")
    else:
        print("⚠️ Data file not found. Please check DATA_FILE_PATH.")
        print(f"Current Pwd: {os.getcwd()}")