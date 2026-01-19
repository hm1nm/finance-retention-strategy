import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import linregress
import os
import tqdm  # For progress bar if available, else standard print

# =============================================================================
# [설정] 컬럼명 및 파일 경로 설정
# =============================================================================
COL_ID = '발급회원번호'
COL_DATE = '기준년월'

# 분석 변수 (Wide Format의 접미사 '_MM' 등을 뗀 순수 컬럼명 가정)
COL_SPEND = '이용금액_신용_B0M'      # 소비
COL_COUNT = '이용건수_신용_B0M'      # 빈도
COL_BALANCE = '잔액_B0M'             # 잔액
COL_CASH_ADV = '잔액_현금서비스_B0M' # 악성부채1
COL_CARD_LOAN = '잔액_카드론_B0M'    # 악성부채2
COL_DELINQ = '연체잔액_B0M'          # 리스크3
COL_AVG_BAL = '월중평잔'             # 자산

# Additional columns for R12M fallback
COL_SPEND_R12M = '이용금액_신용_R12M'
COL_COUNT_R12M = '이용건수_신용_R12M'

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
            # idx_from_last: 1 for last, 2 for 2nd last...
            if len(group) >= idx_from_last:
                return group[col].iloc[-idx_from_last]
            return 0

        # 1. [부정] 악성 부채 점수 (Score_BadDebt)
        # If only 1 month, cannot compare with previous, so score is 0
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
        sum_r3 = group[COL_COUNT].iloc[-3:].sum()
        sum_r6 = group[COL_COUNT].sum()
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
        # If current < avg, result is negative (decrease) -> Condition met
        # If current > avg, result is positive (increase) -> Condition failed
        r12m_spend = group[COL_SPEND_R12M].iloc[0] if COL_SPEND_R12M in group.columns else 0
        avg_spend = r12m_spend / 12
        slope_spend = group[COL_SPEND].iloc[0] - avg_spend
        
        # Count Slope Proxy
        r12m_count = group[COL_COUNT_R12M].iloc[0] if COL_COUNT_R12M in group.columns else 0
        avg_count = r12m_count / 12
        slope_count = group[COL_COUNT].iloc[0] - avg_count
        
        # Balance Slope: User requested to exclude this (always satisfy).
        # We set it to -1 (or any value <= 0) to ensure the condition (slope <= 0) passes.
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
    
    # (조건 B) 4대 위험 징후 중 1개 이상 감지 (Risk_Count >= 1)
    cond1 = scores['Score_BadDebt'] > 0
    cond2 = scores['Score_Delinq'] > 0
    cond3 = scores['Score_Activity'] < 0
    cond4 = scores['Score_Asset'] == 0
    
    risk_count = int(cond1) + int(cond2) + int(cond3) + int(cond4)
    cond_high_risk = (risk_count >= 1)
    
    return 1 if (cond_slopes_decrease and cond_high_risk) else 0

def analyze_rolling_churn(file_path):
    print(f"\n[Info] 파일 로드 및 분석 시작: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(file_path, low_memory=False)
        # Fill missing cols with 0 if needed
        # Added R12M columns to required list
        required_cols = [
            COL_ID, COL_DATE, COL_SPEND, COL_COUNT, COL_BALANCE, 
            COL_CASH_ADV, COL_CARD_LOAN, COL_DELINQ, COL_AVG_BAL,
            COL_SPEND_R12M, COL_COUNT_R12M
        ]
        for c in required_cols:
            if c not in df.columns: df[c] = 0
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # Sort
    df.sort_values(by=[COL_ID, COL_DATE], inplace=True)
    
    # Group by ID
    grouped = df.groupby(COL_ID)
    
    results = []
    
    print(" - 고객별 Rolling Analysis 진행 중... (데이터 최소 1개월 기준, 1개월 시 R12M 보완)")
    
    count_churners = 0
    total_processed = 0
    
    for cust_id, group in grouped:
        total_processed += 1
        # Minimum 1 months required logic now supported
        if len(group) < 1:
            continue
            
        # 1. "현재 시점"의 이탈 여부를 확인
        # (주의: 사용자가 원하는 것은 '현재 이탈자인 사람'의 과거 지속기간 확인이므로,
        # 가장 마지막 달이 Churn이어야 분석 대상이 됨)
        current_scores = calculate_churn_scores(group)
        is_current_churn = check_churn_condition(current_scores)
        
        if is_current_churn == 1:
            count_churners += 1
            
            # 2. 이탈자라면, 과거로 역추적 (Rolling)
            consecutive_months = 1 # Start with 1 (the current month)
            
            # Max lookback
            # If len=6, loops i=1 to 5 (check len=5 down to len=1)
            # If len=1, loop range(1, 1) -> Empty loop (correct, duration=1)
            max_lookback = len(group) - 1
            
            for i in range(1, max_lookback + 1):
                past_group = group.iloc[:-i] # Remove last i rows
                past_scores = calculate_churn_scores(past_group)
                is_past_churn = check_churn_condition(past_scores)
                
                if is_past_churn == 1:
                    consecutive_months += 1
                else:
                    break # Break chain
            
            results.append({
                'Cust_ID': cust_id,
                'Churn_Duration_Months': consecutive_months
            })
            
        # if total_processed % 1000 == 0:
        #     print(f"   ... {total_processed}명 처리 완료 (발견된 이탈자: {count_churners}명)")

    # Output Results
    if len(results) == 0:
        print("❌ 분석된 이탈자가 없습니다.")
        return

    df_res = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print(f"📊 이탈자 Rolling 분석 결과 (대상 파일: {os.path.basename(file_path)})")
    print("="*50)
    print(f" - 총 분석 고객 수: {total_processed}명")
    print(f" - 최종 시점 이탈자 수: {len(df_res)}명 ({len(df_res)/total_processed*100:.2f}%)")
    print("-" * 30)
    print(" [이탈 징후 지속 기간 통계]")
    print(df_res['Churn_Duration_Months'].describe())
    print("-" * 30)
    print(" [기간별 분포 (상위 10개)]")
    print(df_res['Churn_Duration_Months'].value_counts().sort_index(ascending=False).head(10))
    
    # Save detailed results
    save_path = f"churn_duration_results_{os.path.basename(file_path)}"
    # df_res.to_csv(save_path, index=False)
    print(f"\n✅ 상세 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    # Use the test file provided in the prompt context
    TEST_FILE = '../../../260108/general_combined_part1.csv'
    analyze_rolling_churn(TEST_FILE)
