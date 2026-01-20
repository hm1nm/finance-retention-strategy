import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support

def safe_read_csv(path):
    """
    CSV 파일을 안전하게 로드합니다.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"[로드 성공] shape = {df.shape}")
        return df
    except Exception as e:
        print(f"❌ [로드 실패] {e}")
        return None

def generate_target_method_b(df, ratio_threshold=0.8):
    """
    VIP 타겟 생성 (Method B: 20% 감소 룰)
    - Target = 1 (이탈): 당월 총 이용금액 < (직전 3개월 평균 * 0.8)
    - Target = 0 (유지): 그 외
    - 판단 제외(NaN): 직전 3개월 평균이 0/결측인 경우(휴면/신규)
    """
    if df is None:
        return None, None

    required_cols = [
        "발급회원번호", "기준년월",
        "이용금액_신용_B0M", "이용금액_체크_B0M",
        "이용금액_신용_R3M", "이용금액_체크_R3M",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        print("❌ [타겟 생성 실패] 필수 컬럼 누락:", missing)
        return None, None

    work = df.copy()

    # 정렬(회원/월 기준) — 원본 의도 유지
    try:
        work = work.sort_values(by=["발급회원번호", "기준년월"]).copy()
    except Exception as e:
        print("⚠️ [정렬 경고] 정렬 실패:", e)

    # 결측 0 처리(타겟 생성용 컬럼만)
    for c in ["이용금액_신용_B0M", "이용금액_체크_B0M", "이용금액_신용_R3M", "이용금액_체크_R3M"]:
        work[c] = work[c].fillna(0)

    # 당월/직전3M 평균 산출
    work["당월_총_이용금액"] = work["이용금액_신용_B0M"] + work["이용금액_체크_B0M"]
    work["직전_3M_평균_이용금액"] = (work["이용금액_신용_R3M"] + work["이용금액_체크_R3M"]) / 3

    # Target 초기화
    work["Target"] = np.nan

    # 판단 가능 조건: 직전 3M 평균 > 0
    can_judge = work["직전_3M_평균_이용금액"] > 0

    # 비율 계산 및 타겟 부여
    ratio = pd.Series(np.nan, index=work.index)
    ratio.loc[can_judge] = work.loc[can_judge, "당월_총_이용금액"] / work.loc[can_judge, "직전_3M_평균_이용금액"]

    work.loc[can_judge & (ratio < ratio_threshold), "Target"] = 1
    work.loc[can_judge & (ratio >= ratio_threshold), "Target"] = 0

    train_df = work[work["Target"].notna()].copy()
    train_df["Target"] = train_df["Target"].astype(int)

    dormant_df = work[work["Target"].isna()].copy()

    print("\n" + "="*50)
    print("📊 타겟 생성 결과 (Method B: 20% 감소 룰)")
    print("="*50)
    print(f" - 전체 데이터: {len(work)}건")
    print(f" - 학습용 데이터(0/1): {len(train_df)}건")
    if len(train_df) > 0:
        print(f"   └ 이탈(1): {(train_df['Target']==1).sum()}건 ({train_df['Target'].mean()*100:.2f}%)")
        print(f"   └ 유지(0): {(train_df['Target']==0).sum()}건")
    print(f" - 판단 제외(NaN): {len(dormant_df)}건")
    print("="*50)

    return train_df, dormant_df

def prepare_Xy_group_split(train_df, id_col="발급회원번호", target_col="Target", drop_b0m=True):
    """
    모델 입력 데이터 구성 (누수 제거 + 발급회원번호 기준 분리)
    """
    if train_df is None or len(train_df) == 0:
        print("❌ [데이터 준비 실패] 학습용 train_df가 비어있습니다.")
        return None, None, None, None

    if id_col not in train_df.columns or target_col not in train_df.columns:
        print("❌ [데이터 준비 실패] 필수 컬럼 누락:", id_col, target_col)
        return None, None, None, None

    data = train_df.copy()

    # 타겟 생성에 사용된 컬럼(누수 가능) 제거
    leakage_cols = [
        "당월_총_이용금액", "직전_3M_평균_이용금액",
        "이용금액_신용_B0M", "이용금액_체크_B0M",
        "이용금액_신용_R3M", "이용금액_체크_R3M",
    ]

    # B0M 포함 컬럼은 당월 정보이므로 통째로 제거(원본 의도)
    b0m_cols = []
    if drop_b0m:
        b0m_cols = [c for c in data.columns if "B0M" in c]

    drop_cols = list(set([id_col, "기준년월", target_col] + leakage_cols + b0m_cols))
    drop_cols_exist = [c for c in drop_cols if c in data.columns]

    X = data.drop(columns=drop_cols_exist, errors="ignore").copy()
    y = data[target_col].astype(int).copy()

    # 회원 단위로 stratify (회원별 target 1개만 추출)
    unique_ids = data[[id_col, target_col]].drop_duplicates(subset=[id_col]).copy()
    try:
        train_ids, test_ids = train_test_split(
            unique_ids[id_col],
            test_size=0.2,
            random_state=42,
            stratify=unique_ids[target_col]
        )
    except Exception as e:
        # stratify 불가능(소수 클래스가 너무 적거나 등) 시 fallback
        print("⚠️ stratify split 실패 → 비층화 split로 대체:", e)
        train_ids, test_ids = train_test_split(
            unique_ids[id_col],
            test_size=0.2,
            random_state=42
        )

    train_mask = data[id_col].isin(train_ids)
    test_mask = data[id_col].isin(test_ids)

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    # 그룹 누수 체크
    overlap = set(data.loc[train_mask, id_col]).intersection(set(data.loc[test_mask, id_col]))
    print("[그룹 누수 체크] 겹치는 회원 수 =", len(overlap))

    print("\n[데이터 준비 완료]")
    print(" - X_train:", X_train.shape, " y_train:", y_train.shape, " (이탈비중:", round(y_train.mean(), 4), ")")
    print(" - X_test :", X_test.shape,  " y_test :", y_test.shape,  " (이탈비중:", round(y_test.mean(), 4), ")")

    return X_train, X_test, y_train, y_test

def encode_and_fillna(X_train, X_test):
    """
    범주형 처리(LabelEncoder) + 결측 처리
    """
    if X_train is None:
        return None, None

    Xtr = X_train.copy()
    Xte = X_test.copy()

    cat_cols = Xtr.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        try:
            le = LabelEncoder()
            full = pd.concat([Xtr[col], Xte[col]], axis=0).astype(str)
            le.fit(full)
            Xtr[col] = le.transform(Xtr[col].astype(str))
            Xte[col] = le.transform(Xte[col].astype(str))
        except Exception as e:
            # 해당 컬럼에서 인코딩이 문제가 나면 과감히 제거(에러 방지)
            print(f"⚠️ [인코딩 실패] 컬럼 제거: {col} / 사유: {e}")
            Xtr = Xtr.drop(columns=[col], errors="ignore")
            Xte = Xte.drop(columns=[col], errors="ignore")

    # 결측치 채우기
    Xtr = Xtr.fillna(-999)
    Xte = Xte.fillna(-999)

    print("[인코딩/결측 처리 완료] X_train:", Xtr.shape, "X_test:", Xte.shape)
    return Xtr, Xte

def run_random_forest(X_train, X_test, y_train, y_test):
    """
    RandomForest 학습 + 평가
    """
    if X_train is None or X_test is None:
        print("❌ [RF 실행 불가] 입력 데이터가 None 입니다.")
        return None

    if len(np.unique(y_train)) < 2:
        print("❌ [RF 실행 불가] y_train에 클래스가 1개뿐입니다. (이탈/유지 둘 다 있어야 함)")
        return None

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    print("\n[RandomForest 학습 시작...]")
    model.fit(X_train, y_train)

    print("[학습 완료, 예측 수행 중...]")
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred)
    print("\n[RandomForest Confusion Matrix]\n", cm)

    print("\n[Classification Report]")
    print(classification_report(y_test, pred, digits=4))

    try:
        auc = roc_auc_score(y_test, proba)
        print("[ROC-AUC] =", round(auc, 4))
    except Exception as e:
        auc = None
        print("⚠️ ROC-AUC 계산 실패:", e)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    print(f"[요약] Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | AUC={auc}")

    result = {
        "model": model,
        "y_proba": proba,
        "y_pred": pred,
        "confusion_matrix": cm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }
    return result

def optimize_threshold(res_rf, y_test):
    """
    Threshold 튜닝 (Recall 중심)
    """
    if res_rf is None:
        print("⚠️ RF 결과가 없어서 Threshold 튜닝을 생략합니다.")
        return

    proba = res_rf["y_proba"]
    thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        rows.append([t, prec, rec, f1])

    df_thr = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])
    print("\n[Threshold 튜닝 결과 (Top 15)]")
    print(df_thr.sort_values(by=["recall", "precision"], ascending=False).head(15))

def main():
    parser = argparse.ArgumentParser(description="VIP Random Forest Model Pipeline")
    parser.add_argument("--data_path", type=str, default="../../DATA/VIP/VIP_combined_part0.csv", help="Input CSV file path")
    parser.add_argument("--ratio_threshold", type=float, default=0.8, help="Ratio threshold for target generation")
    
    args = parser.parse_args()

    print(f"[시작] 데이터 경로: {args.data_path}")
    
    # 1. 데이터 로드
    df = safe_read_csv(args.data_path)
    if df is None:
        return

    # 2. 타겟 생성
    train_data, dormant_data = generate_target_method_b(df, ratio_threshold=args.ratio_threshold)
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = prepare_Xy_group_split(train_data)

    # 4. 전처리 (인코딩/결측)
    X_train_enc, X_test_enc = encode_and_fillna(X_train, X_test)

    # 5. 모델 학습 및 평가
    res_rf = run_random_forest(X_train_enc, X_test_enc, y_train, y_test)

    # 6. Threshold 튜닝
    if res_rf is not None:
        optimize_threshold(res_rf, y_test)

if __name__ == "__main__":
    main()
