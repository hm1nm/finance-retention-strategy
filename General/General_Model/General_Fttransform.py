import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import linregress
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import platform
import time

# ========================================================================================
# 1. Configuration & Constants
# ========================================================================================

# 한글 폰트 설정
if platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Windows
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False

COL_ID = '발급회원번호'
COL_DATE = '기준년월'

# [수정됨] 실제 데이터 명세서(필드한글명) 기반 매핑
COL_SPEND = '이용금액_신용_B0M'
COL_COUNT = '이용건수_신용_B0M'
COL_BALANCE = '잔액_B0M'
COL_AVG_BAL = '평잔_3M'
COL_CASH_ADV = '이용금액_CA_B0M'
COL_CARD_LOAN = '이용금액_카드론_B0M'
COL_DELINQ = '회원여부_연체'
COL_SPEND_R12M = '이용금액_신용_R12M'
COL_COUNT_R12M = '이용건수_신용_R12M'

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================================================================================
# 2. Utility Functions
# ========================================================================================
def calc_slope_long(series):
    y = series.values
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n)
    if np.all(y == y[0]): return 0.0
    slope, _, _, _, _ = linregress(x, y)
    if np.isnan(slope): return 0.0
    return slope

def normalize_risk_vector(series):
    if series.empty: return series
    risk_raw = -series
    min_val = risk_raw.min()
    max_val = risk_raw.max()
    if max_val == min_val: return pd.Series(0, index=series.index)
    normalized = (risk_raw - min_val) / (max_val - min_val)
    return normalized

def calculate_vif(dataframe, sample_size=5000):
    print("\n🔍 [VIF Check] Calculating Variance Inflation Factors...")
    df_vif_input = dataframe.select_dtypes(include=[np.number]).dropna()
    if len(df_vif_input) > sample_size:
        df_vif_input = df_vif_input.sample(n=sample_size, random_state=42)
    
    cols_to_exclude = ['Target', '발급회원번호', 'Unnamed: 0', 'index']
    cols_check = [c for c in df_vif_input.columns if c not in cols_to_exclude]
    df_vif_input = df_vif_input[cols_check]
    df_vif_input = add_constant(df_vif_input)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_vif_input.columns
    try:
        vif_data["VIF"] = [variance_inflation_factor(df_vif_input.values, i) for i in range(df_vif_input.shape[1])]
    except Exception as e:
        print(f"⚠️ VIF calculation failed: {e}")
        return None

    vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values(by="VIF", ascending=False)
    print(vif_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="VIF", y="Feature", data=vif_data.head(20))
    plt.title("Top 20 Features by VIF")
    plt.tight_layout()
    plt.show()
    return vif_data

# ========================================================================================
# 3. Core Logic: Scoring & Target Generation
# ========================================================================================
def calculate_churn_scores(group):
    # (Same logic as existing pipelines)
    months_data = len(group)
    if months_data >= 2:
        slope_spend = calc_slope_long(group[COL_SPEND])
        slope_balance = calc_slope_long(group[COL_BALANCE])
        slope_count = calc_slope_long(group[COL_COUNT])
    else:
        current_spend = group[COL_SPEND].iloc[-1]
        r12m_spend = group[COL_SPEND_R12M].iloc[-1] if COL_SPEND_R12M in group.columns else 0
        slope_spend = current_spend - r12m_spend
        current_count = group[COL_COUNT].iloc[-1]
        r12m_count = group[COL_COUNT_R12M].iloc[-1] if COL_COUNT_R12M in group.columns else 0
        slope_count = current_count - r12m_count
        slope_balance = -1 
        
    score_status_total = 0
    delinq_sum = group[COL_DELINQ].sum() if COL_DELINQ in group.columns else 0
    cash_adv_sum = group[COL_CASH_ADV].sum() if COL_CASH_ADV in group.columns else 0

    if delinq_sum > 0: score_status_total += 50
    if cash_adv_sum > 0: score_status_total += 30
    
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
    print(" - Generating Target Variable...")
    cond_slopes_decrease = ((df_scores['Slope_Spend'] <= 0) & (df_scores['Slope_Balance'] <= 0) & (df_scores['Slope_Count'] <= 0))
    cond1 = df_scores['Score_BadDebt'] > 0
    cond2 = df_scores['Score_Delinq'] > 0
    cond3 = df_scores['Score_Activity'] < 0
    cond4 = df_scores['Score_Asset'] == 0 
    risk_count = cond1.astype(int) + cond2.astype(int) + cond3.astype(int) + cond4.astype(int)
    cond_high_risk = (risk_count >= 1)
    
    df_scores['Target'] = np.where(cond_slopes_decrease & cond_high_risk, 1, 0)
    
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
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None
        
    if COL_SPEND_R12M not in df.columns: df[COL_SPEND_R12M] = 0
    if COL_COUNT_R12M not in df.columns: df[COL_COUNT_R12M] = 0

    print(" - Sorting data...")
    df.sort_values(by=[COL_ID, COL_DATE], inplace=True)
    
    print("2. [Scoring] Calculating Churn Scores...")
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

def plot_score_distributions(df, target_col='Target'):
    print("\n📊 [Distribution Analysis] Plotting Score Distributions...")
    cols = ['Final_Total_Score', 'Score_Slope_Total', 'Score_Status_Total', 'Slope_Spend', 'Slope_Count', 'Slope_Balance']
    cols = [c for c in cols if c in df.columns]
    if not cols: return
    fig, axes = plt.subplots(nrows=len(cols), ncols=2, figsize=(15, 4 * len(cols)))
    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, element="step", ax=axes[i, 0], palette='Set1')
        axes[i, 0].set_title(f'{col} Distribution by Target')
        sns.boxplot(data=df, x=target_col, y=col, ax=axes[i, 1], palette='Set1')
        axes[i, 1].set_title(f'{col} Boxplot by Target')
    plt.tight_layout()
    plt.show()

# ========================================================================================
# 4. FT-Transformer Implementation
# ========================================================================================

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class FeatureTokenizer(nn.Module):
    def __init__(self, num_numerical_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_numerical_features, d_token))
        self.bias = nn.Parameter(torch.randn(num_numerical_features, d_token))
        
    def forward(self, x_num):
        # x_num: (batch_size, num_numerical_features)
        # out: (batch_size, num_numerical_features, d_token)
        x = x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return x

class FTTransformer(nn.Module):
    def __init__(self, num_numerical_features, d_token=192, n_layers=3, n_heads=8, d_ffn=None, attention_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = d_token * 4 // 3
            
        self.tokenizer = FeatureTokenizer(num_numerical_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, 
            nhead=n_heads, 
            dim_feedforward=d_ffn, 
            dropout=attention_dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )
        
    def forward(self, x_num):
        # x_num: (batch_size, num_features)
        batch_size = x_num.shape[0]
        
        # Tokenize (Embeddings)
        x = self.tokenizer(x_num) # (B, F, D)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1) # (B, F+1, D)
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for prediction
        x_cls = x[:, 0, :]
        logits = self.head(x_cls)
        return logits.squeeze(-1)

# ========================================================================================
# 5. Training Logic
# ========================================================================================

def train_eval_fttransformer(df_train, target_col='Target'):
    print(f"\n🏋️ [Model Training] FT-Transformer (PyTorch)...")
    
    # Feature Selection (Removing Leakage)
    leakage_cols = [
        target_col, COL_ID, 'Unnamed: 0',
        'Slope_Spend', 'Slope_Balance', 'Slope_Count',
        'Score_BadDebt', 'Score_Delinq', 'Score_Activity', 'Score_Asset',
        'Score_Status_Total', 'Score_Slope_Total', 'Final_Total_Score'
    ]
    
    continuous_features = [c for c in df_train.columns if c not in leakage_cols and pd.api.types.is_numeric_dtype(df_train[c])]
    
    print(f" - Using {len(continuous_features)} numerical features.")
    
    X = df_train[continuous_features]
    y = df_train[target_col]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scaling (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=continuous_features)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=continuous_features)
    
    # Dataset & DataLoader
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    model = FTTransformer(
        num_numerical_features=len(continuous_features),
        d_token=192,
        n_layers=3,
        n_heads=8,
        attention_dropout=0.2
    ).to(device)
    
    # Class Weight needed for imbalanced data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device) if pos_count > 0 else torch.tensor([1.0], device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training Loop
    epochs = 30
    best_auc = 0
    patience = 5
    counter = 0
    
    print(f" - Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_auc = roc_auc_score(all_targets, all_preds)
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_ft_transformer.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("   Early stopping triggered.")
                break
                
    # Final Evaluation
    model.load_state_dict(torch.load('best_ft_transformer.pth'))
    model.eval()
    
    y_pred_probs = []
    y_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            y_pred_probs.extend(probs.cpu().numpy())
            y_true.extend(y_batch.numpy())
            
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    y_true = np.array(y_true)
    
    print("\n[FT-Transformer Results]")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_labels):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_labels):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred_probs):.4f}")
    print(classification_report(y_true, y_pred_labels))
    
    # Visualizations
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - FT-Transformer')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Permutation Importance (Simple)
    print(" - Calculating Permutation Importance...")
    calculate_permutation_importance(model, X_test, y_true, device)
    
    return model

def calculate_permutation_importance(model, X_df, y_true, device):
    """
    Calculate simple permutation importance for top features.
    """
    baseline_auc = roc_auc_score(y_true, predict_proba(model, X_df, device))
    importances = {}
    
    # Check top features by variance or simple correlation to speed up if needed
    # checking all for now
    features = X_df.columns
    
    for col in features:
        save = X_df[col].copy()
        X_df[col] = np.random.permutation(X_df[col])
        
        permuted_auc = roc_auc_score(y_true, predict_proba(model, X_df, device))
        importances[col] = baseline_auc - permuted_auc
        
        X_df[col] = save # restore
        
    imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
    plt.title('Permutation Feature Importance (FT-Transformer)')
    plt.tight_layout()
    plt.show()

def predict_proba(model, X_df, device):
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs)
    return probs.cpu().numpy()

# ========================================================================================
# 6. Main Execution
# ========================================================================================
DATA_FILE_PATH = './260108/general_combined_part1.csv'

if __name__ == "__main__":
    print("🚀 Starting Feature Selection & Visualization Pipeline (FT-Transformer Version)...")
    
    if os.path.exists(DATA_FILE_PATH):
        df_final = process_data_and_merge(DATA_FILE_PATH)
        
        if df_final is not None:
            # VIF and Distribution check
            plot_score_distributions(df_final)
            # define train_eval_fttransformer
            model = train_eval_fttransformer(df_final)
            
            print("✅ Pipeline (FT-Transformer) Completed Successfully.")
    else:
        print("⚠️ Data file not found. Please check DATA_FILE_PATH.")
        print(f"Current Pwd: {os.getcwd()}")
