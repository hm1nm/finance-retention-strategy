import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_lightning.callbacks import Callback

# PyTorch Tabular 임포트
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
except ImportError:
    print("Error: 'pytorch_tabular' 라이브러리가 설치되어 있지 않습니다.")
    print("pip install pytorch_tabular[all] 명령어로 설치해주세요.")
    sys.exit(1)

# 경고 무시 및 설정
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')


def load_data(file_path):
    """
    CSV 데이터를 불러옵니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다. ({file_path})")
        return None

    try:
        # 인코딩 문제 대응
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
            
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

    print("타겟 정의 중 (Method B strict)...")
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
        '이용금액_신용_B0M', '이용금액_체크_B0M', '이용금액_신용_R3M', '이용금액_체크_R3M',
        '이용금액_신용_BOM', '이용금액_체크_BOM' # 오타 등 포함
    ]
    danger_keywords = ['B0M', '당월', '금액', '횟수', '이용', '여부', '건수', '신청']
    auto_leakage = [col for col in train_df.columns if any(word in col for word in danger_keywords)]
    base_drop = ['발급회원번호', '기준년월'] # 타겟은 나중에 분리

    final_drop_set = set(leakage_cols + auto_leakage + base_drop)
    
    # ID 기반 Split
    unique_ids = train_df['발급회원번호'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    train_final = train_df[train_df['발급회원번호'].isin(train_ids)].copy()
    test_final = train_df[train_df['발급회원번호'].isin(test_ids)].copy()
    
    # 컬럼 필터링 (Feature Selection)
    feature_columns = [c for c in train_df.columns if c not in final_drop_set and c != target_col]
    
    # 실제 존재하는 컬럼만 선택 + 타겟 컬럼 포함 (PyTorch Tabular는 DataFrame 전체를 입력으로 받음)
    use_cols = feature_columns + [target_col]
    
    train_final = train_final[use_cols].reset_index(drop=True)
    test_final = test_final[use_cols].reset_index(drop=True)

    # 결측치/무한대 처리
    train_final = train_final.replace([np.inf, -np.inf], 0).fillna(0)
    test_final = test_final.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"최종 Train 크기: {train_final.shape}, Test 크기: {test_final.shape}")
    print(f"사용 피처 수: {len(feature_columns)}")
    
    return train_final, test_final, feature_columns, target_col

def apply_scaling(train_df, test_df, feature_cols):
    """
    수치형 변수 표준화 (Standard Scaling)
    """
    # 수치형/범주형 자동 분류
    cat_cols = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    scaler = StandardScaler()
    if num_cols:
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        print(f"수치형 변수 {len(num_cols)}개 표준화 완료")
        
    return train_df, test_df, num_cols, cat_cols

# 가중치 주입 콜백
class SafeLossWeightCallback(Callback):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def on_train_start(self, trainer, pl_module):
        device = pl_module.device
        # Loss 함수를 가중치가 적용된 CrossEntropyLoss로 교체
        if not isinstance(pl_module.loss, nn.CrossEntropyLoss) or pl_module.loss.weight is None:
            pl_module.loss = nn.CrossEntropyLoss(weight=self.weights.to(device))

def train_ft_transformer(train_df, test_df, num_cols, cat_cols, target_col, output_dir):
    """
    FT-Transformer 모델 설정 및 학습
    """
    # 1. 클래스 가중치 계산
    y_train = train_df[target_col].values
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"클래스 가중치: {weights}")

    # 2. Config 설정
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=num_cols,
        categorical_cols=cat_cols,
    )

    model_config = FTTransformerConfig(
        task="classification",
        num_heads=8,
        num_attn_blocks=4,
        learning_rate=1e-5,
        head="LinearHead",
        head_config={"layers": "128-64", "dropout": 0.1},
    )

    trainer_config = TrainerConfig(
        batch_size=64,
        max_epochs=15,
        early_stopping_patience=3,
        gradient_clip_val=0.5,
        checkpoints="valid_loss",
        checkpoints_path=os.path.join(output_dir, "checkpoints"),
        load_best=True
    )

    # 3. 모델 초기화
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=OptimizerConfig(),
        trainer_config=trainer_config,
    )

    # 4. 학습
    print("모델 학습 시작 (FT-Transformer)...")
    tabular_model.fit(
        train=train_df,
        validation=test_df,
        callbacks=[SafeLossWeightCallback(weights_tensor)]
    )
    
    return tabular_model

def evaluate_and_save(model, test_df, target_col, output_dir):
    """
    모델 평가 및 결과 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 저장
    model.save_model(os.path.join(output_dir, "ft_transformer_model"))
    
    # 예측
    pred_df = model.predict(test_df)
    pred_col = f"{target_col}_prediction"
    
    y_true = test_df[target_col]
    y_pred = pred_df[pred_col]
    
    # 리포트 출력
    report = classification_report(y_true, y_pred)
    print(f"\n--- Classification Report ---\n{report}")
    
    # 파일 저장
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # 혼동 행렬 이미지 저장
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\nTarget: {target_col}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"결과 저장 완료: {output_dir}")

def main(file_path, output_dir="FT_Transformer_Result"):
    # 1. 데이터 로드
    df = load_data(file_path)
    if df is None: return

    # 2. 전처리 및 타겟 정의
    train_df = preprocess_and_define_target(df)
    
    # 3. Leakage 제거 및 Split
    train_final, test_final, feature_cols, target_col = remove_leakage_and_split(train_df)
    
    # 4. 스케일링
    train_final, test_final, num_cols, cat_cols = apply_scaling(train_final, test_final, feature_cols)
    
    # 5. 학습
    model = train_ft_transformer(train_final, test_final, num_cols, cat_cols, target_col, output_dir)
    
    # 6. 저장 및 평가
    evaluate_and_save(model, test_final, target_col, output_dir)

if __name__ == "__main__":
    # 실행 예시
    FILE_PATH = "../../DATA/VIP/VIP/VIP_combined_part1.csv"
    OUTPUT_DIR = "./FT_Transformer_Out"
    
    if os.path.exists(FILE_PATH):
        main(FILE_PATH, OUTPUT_DIR)
    else:
        print(f"Warning: '{FILE_PATH}' 파일이 없습니다.")
        print("스크립트 하단의 FILE_PATH를 수정하거나 main() 함수를 직접 호출하세요.")
