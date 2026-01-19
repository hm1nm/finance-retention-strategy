import pandas as pd
import numpy as np
import sys
import os

def load_and_sort_data(file_path):
    """
    CSV 파일을 로드하고 발급회원번호와 기준년월 기준으로 정렬합니다.
    """
    try:
        # low_memory=False 옵션은 DtypeWarning 방지를 위해 추가
        df = pd.read_csv(file_path, low_memory=False)
        df_sorted = df.sort_values(by=['발급회원번호', '기준년월']).copy()
        return df_sorted
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다 - {file_path}")
        sys.exit(1)

def calculate_usage_amounts(df):
    """
    당월 총 이용금액과 직전 3개월 평균 이용금액을 계산합니다.
    """
    df = df.copy()
    # 당월 총 이용금액 (신용 + 체크)
    # 컬럼 존재 여부 체크는 생략되었으나 필요 시 추가 가능
    df['당월_총_이용금액'] = df['이용금액_신용_B0M'] + df['이용금액_체크_B0M']
    
    # 직전 3개월 평균 이용금액 (신용 R3M + 체크 R3M) / 3
    df['직전_3M_평균_이용금액'] = (df['이용금액_신용_R3M'] + df['이용금액_체크_R3M']) / 3
    
    return df

def _define_churn_logic(row, threshold_ratio=0.8):
    """
    개별 행에 대해 이탈 여부를 판단하는 로직입니다.
    
    Args:
        row: 데이터프레임의 행
        threshold_ratio: 이탈 판단 임계 비율 (기본값 0.8, 즉 80%)
        
    Returns:
        1: 이탈 (직전 3개월 평균 대비 threshold_ratio 미만 사용)
        0: 유지
        np.nan: 판단 제외 (직전 3개월 평균이 없거나 0원인 경우)
    """
    # 직전 3개월 평균이 없거나 0원인 경우 -> '판단 제외(휴면/신규)'로 분류
    if pd.isna(row['직전_3M_평균_이용금액']) or row['직전_3M_평균_이용금액'] <= 0:
        return np.nan 
    
    # 직전 평균 대비 설정 비율 미만 사용 시 이탈(1), 아니면 유지(0)
    if row['당월_총_이용금액'] < (row['직전_3M_평균_이용금액'] * threshold_ratio):
        return 1
    else:
        return 0

def generate_target_data(df, threshold_ratio=0.8):
    """
    데이터프레임에 이탈 타겟 레이블을 생성합니다.
    """
    df = df.copy()
    # apply 함수를 사용하여 행별로 로직 적용
    df['이탈_타겟'] = df.apply(lambda row: _define_churn_logic(row, threshold_ratio), axis=1)
    return df

def split_dataset(df):
    """
    타겟 유무에 따라 학습용 데이터와 판단 제외 데이터로 분리합니다.
    
    Returns:
        train_df: 학습용 데이터 (타겟 0, 1)
        dormant_new_df: 판단 제외 데이터 (타겟 NaN)
    """
    # 타겟이 0 또는 1로 확실히 정해진 데이터 (학습용)
    train_df = df[df['이탈_타겟'].notna()].copy()
    train_df['이탈_타겟'] = train_df['이탈_타겟'].astype(int)

    # 타겟이 NaN인 데이터 (별도 관리용: 장기 휴면 또는 신규 고객)
    dormant_new_df = df[df['이탈_타겟'].isna()].copy()
    
    return train_df, dormant_new_df

def print_result_summary(train_df, dormant_new_df):
    """
    데이터 분리 결과를 출력합니다.
    """
    print("--- [데이터 분리 결과] ---")
    print(f"1. 학습 가능 데이터(0, 1): {len(train_df)}건")
    print(f"   - 유지(0): {len(train_df[train_df['이탈_타겟'] == 0])}건")
    print(f"   - 이탈(1): {len(train_df[train_df['이탈_타겟'] == 1])}건")
    print(f"\n2. 판단 제외 데이터(NaN): {len(dormant_new_df)}건 (장기 미사용/신규)")
    print("--------------------------")

def main(input_file, output_train=None, output_dormant=None, threshold_ratio=0.8):
    """
    전체 프로세스를 실행하는 메인 함수입니다.
    
    Args:
        input_file: 입력 CSV 파일 경로
        output_train: 학습용 데이터 저장 경로 (Optional)
        output_dormant: 제외 데이터 저장 경로 (Optional)
        threshold_ratio: 이탈 판단 기준 비율
    """
    print(f"데이터 로드 중: {input_file}")
    df = load_and_sort_data(input_file)
    
    print("이용금액 계산 중...")
    df = calculate_usage_amounts(df)
    
    print(f"타겟 데이터 생성 중 (Threshold: {threshold_ratio*100}%)...")
    df = generate_target_data(df, threshold_ratio)
    
    train_df, dormant_new_df = split_dataset(df)
    
    print_result_summary(train_df, dormant_new_df)
    
    if output_train:
        train_df.to_csv(output_train, index=False)
        print(f"학습용 데이터 저장 완료: {output_train}")
        
    if output_dormant:
        dormant_new_df.to_csv(output_dormant, index=False)
        print(f"판단 제외 데이터 저장 완료: {output_dormant}")

if __name__ == "__main__":
    # 스크립트로 직접 실행 시 예시 설정
    INPUT_CSV = "GENERAL_30K_merged_data_240636_822.csv"
    
    if os.path.exists(INPUT_CSV):
        main(INPUT_CSV)
    else:
        print(f"'{INPUT_CSV}' 파일을 찾을 수 없습니다. (현재 디렉토리: {os.getcwd()})")
        print("모듈로 import해서 사용하거나 파일 경로를 확인해주세요.")
