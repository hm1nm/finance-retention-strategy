import os
import gc
import glob
import time
import joblib
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Suppress Warnings
warnings.filterwarnings('ignore')

# Set Korean Font
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)


# =============================================================================
# 1. Configuration
# =============================================================================
class Config:
    # Paths
    BASE_PATH = "c:/Users/johnh/Documents/Test_code/finance-retention-strategy/"
    KEY_FILE = os.path.join(BASE_PATH, "GENERAL22/상태4종 이진분류 기울기/General_Model_Input_Ids.csv") # Placeholder path, needs verification
    DATA_PATH = os.path.join(BASE_PATH, "Data/") # Placeholder, adjust to actual CSV location
    
    # Columns
    COL_ID = '발급회원번호'
    COL_DATE = '기준년월'
    
    # Target definition thresholds
    # (These constants used to be scattered in the notebook)
    RISK_COUNT_THRESHOLD = 2
    
    # Slope Columns
    COLS_SPEND = ['이용금액_신용_B0M', '이용금액_신용_B1M', '이용금액_신용_B2M', '이용금액_신용_B3M', '이용금액_신용_B4M', '이용금액_신용_B5M']
    COLS_COUNT = ['이용건수_신용_B0M', '이용건수_신용_B1M', '이용건수_신용_B2M', '이용건수_신용_B3M', '이용건수_신용_B4M', '이용건수_신용_B5M']
    COLS_BALANCE = ['잔액_B0M', '잔액_B1M', '잔액_B2M', '잔액_B3M', '잔액_B4M', '잔액_B5M']
    
    # Model Hyperparameters (Example from notebook)
    LGBM_PARAMS = {
        'n_estimators': 3000,
        'learning_rate': 0.0156994,
        'max_depth': 11,
        'num_leaves': 100,
        'min_child_samples': 48,
        'subsample': 0.6033,
        'colsample_bytree': 0.7225,
        'reg_alpha': 1.6385e-05,
        'reg_lambda': 3.1678,
        'random_state': 42,
        'n_jobs': -1,
        'device_type': 'cpu' # Change to 'gpu' if available
    }

# =============================================================================
# 2. Utility Functions
# =============================================================================
def clean_mem():
    """Run garbage collection."""
    gc.collect()

def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB --> {end_mem:.2f} MB (Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df

# Helper for parallel processing (Must be top-level for joblib)
def process_file_cpu(file_path, key_ids):
    """
    Load a single CSV file, filter by key_ids, and return DataFrame.
    This function is intended to be used with joblib.Parallel.
    """
    try:
        # 1. Read only header to check columns
        header_df = pd.read_csv(file_path, nrows=0, encoding='utf-8')
        cols = header_df.columns.tolist()
        
        # Check if ID column exists
        if Config.COL_ID not in cols:
            # Try cp949 encoding if utf-8 fails or column missing (logic from notebook)
             header_df = pd.read_csv(file_path, nrows=0, encoding='cp949')
             cols = header_df.columns.tolist()
        
        if Config.COL_ID not in cols:
            return None # Skip if no ID column
            
        # 2. Read file - optimization: usecols if known, here we read all and filter
        # In a real scenario, passing `usecols` would be better if we knew exactly which features
        df = pd.read_csv(file_path, encoding='utf-8') # Default utf-8
        
        # 3. Filter by IDs
        df = df[df[Config.COL_ID].isin(key_ids)]
        
        if df.empty:
            return None
            
        # 4. Memory Reduction
        df = reduce_mem_usage(df)
        
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# =============================================================================
# 3. Data Loader
# =============================================================================
class DataLoader:
    def __init__(self, key_file_path=Config.KEY_FILE, data_dir=Config.DATA_PATH):
        self.key_file_path = key_file_path
        self.data_dir = data_dir
        
    def load_base_ids(self):
        """Load the 'bones' (ID, Date) for the dataset."""
        print(f"Loading key file: {self.key_file_path}")
        # Placeholder logic: In the notebook, this seemed to be `df_main`
        # Using a dummy creation if file not found for demonstration
        if os.path.exists(self.key_file_path):
            df_key = pd.read_csv(self.key_file_path)
            return df_key
        else:
            print("Warning: Key file not found. Creating dummy data for testing.")
            return pd.DataFrame({Config.COL_ID: range(100), Config.COL_DATE: [202201]*100})

    def load_and_merge_features(self, df_base):
        """
        Load feature files in parallel and merge with df_base.
        """
        key_ids = df_base[Config.COL_ID].unique()
        
        # Identify files to load - logic from notebook used glob
        # files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        # For this script, we assume 'dfs_to_concat' logic from notebook
        
        print("Starting parallel data loading...")
        # (Simplified for script: In real usage, uncomment and adjust glob pattern)
        # results = joblib.Parallel(n_jobs=-1)(
        #     joblib.delayed(process_file_cpu)(f, key_ids) for f in files
        # )
        # results = [r for r in results if r is not None]
        
        # Merge logic
        # if results:
        #    df_features = pd.concat(results, axis=1) # Note: Axis=1 concat needs careful index alignment
        #    df_final = pd.merge(df_base, df_features, on=[Config.COL_ID, Config.COL_DATE], how='left')
        #    return df_final
        
        print("Data Loading placeholder - returning base df")
        return df_base

# =============================================================================
# 4. Feature Engineering
# =============================================================================
class FeatureEngineer:
    
    @staticmethod
    def calc_slope(y_values):
        """Calculate slope using scipy linregress (CPU version)."""
        if len(y_values) < 2:
            return 0
        x = np.arange(len(y_values))
        slope, _, _, _, _ = linregress(x, y_values)
        return slope
    
    @staticmethod
    def calculate_slopes(df):
        """Calculate slopes for Spend, Count, Balance."""
        print("Calculating Slopes...")
        
        # Note: Vectorized or GPU approach is preferred for large data.
        # This implementation uses apply for simplicity and compatibility.
        
        # Spend Slope
        # Fill NA with 0 for calculation
        df_spend = df[Config.COLS_SPEND].fillna(0)
        df['Slope_Spend'] = df_spend.apply(lambda row: FeatureEngineer.calc_slope(row.values), axis=1)
        
        # Count Slope
        df_count = df[Config.COLS_COUNT].fillna(0)
        df['Slope_Count'] = df_count.apply(lambda row: FeatureEngineer.calc_slope(row.values), axis=1)
        
        # Balance Slope
        df_balance = df[Config.COLS_BALANCE].fillna(0)
        df['Slope_Balance'] = df_balance.apply(lambda row: FeatureEngineer.calc_slope(row.values), axis=1)
        
        # Normalize slopes (MinMax) - Logic from notebook
        scaler = MinMaxScaler()
        for col in ['Slope_Spend', 'Slope_Count', 'Slope_Balance']:
             # Only normalize if column is not empty/constant
             if df[col].std() > 0:
                 df[f'Norm_{col}'] = scaler.fit_transform(df[[col]])
             else:
                 df[f'Norm_{col}'] = 0
                 
        return df

    @staticmethod
    def calculate_risk_scores(df):
        """Calculate Score_BadDebt, Score_Delinq, etc."""
        print("Calculating Risk Scores...")
        
        # Placeholder logic based on notebook description
        # Example logic: Score = (Current - Avg_3M) / (Avg_3M + epsilon)
        # Notebook logic was more complex involving vectorized changes.
        
        # Here we create dummy scores if columns don't exist, to ensure pipeline runs
        if 'Score_BadDebt' not in df.columns: df['Score_BadDebt'] = 0
        if 'Score_Delinq' not in df.columns: df['Score_Delinq'] = 0
        if 'Score_Activity' not in df.columns: df['Score_Activity'] = 0
        if 'Score_Asset' not in df.columns: df['Score_Asset'] = 0
        
        return df

    @staticmethod
    def assign_churn_segment(df):
        """Classify users into churn segments."""
        print("Assigning Churn Segments...")
        
        # Conditions
        cond1 = df['Score_BadDebt'] > 0
        cond2 = df['Score_Delinq'] > 0
        cond3 = df['Score_Activity'] < 0
        cond4 = df['Score_Asset'] == 0
        
        # Risk Count
        risk_flags = pd.concat([cond1, cond2, cond3, cond4], axis=1)
        df['Risk_Count'] = risk_flags.sum(axis=1)
        
        # Segment Assignment (Priority based)
        df['Churn_Segment'] = '4. 단순 감소/유지군'
        df.loc[cond3, 'Churn_Segment'] = '3. 활동성 급감군'
        df.loc[cond4, 'Churn_Segment'] = '2. 이탈 완료 의심군'
        df.loc[cond1 | cond2, 'Churn_Segment'] = '1. 부실/연체 위험군'
        
        return df

# =============================================================================
# 5. Target Generator
# =============================================================================
class TargetGenerator:
    @staticmethod
    def generate_target(df):
        """
        Target = 1 if (All Slopes <= 0) AND (Risk Count >= 2)
        """
        print("Generating Target...")
        
        slope_condition = (
            (df['Slope_Spend'] <= 0) & 
            (df['Slope_Balance'] <= 0) & 
            (df['Slope_Count'] <= 0)
        )
        
        risk_condition = (df['Risk_Count'] >= Config.RISK_COUNT_THRESHOLD)
        
        df['Target'] = 0
        df.loc[slope_condition & risk_condition, 'Target'] = 1
        
        print(f"Target Distribution:\n{df['Target'].value_counts()}")
        return df

# =============================================================================
# 6. Model Trainer
# =============================================================================
class ModelTrainer:
    def __init__(self, params=Config.LGBM_PARAMS):
        self.params = params
        self.model = None
        
    def train(self, df, feature_cols, target_col='Target'):
        print(f"Starting Training with {len(feature_cols)} features...")
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # LightGBM Dataset
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        
        # Train
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dtest],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Predict
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")
        print("-" * 30)
        print(classification_report(y_test, y_pred))
        
        # Plot Importance
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(self.model, max_num_features=20, title='Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved to 'feature_importance.png'")
        
        return self.model, (acc, f1, auc)

# =============================================================================
# 7. Main Pipeline
# =============================================================================
def main():
    print("=== General Model Pipeline Started ===")
    start_time = time.time()
    
    # 1. Load Data
    loader = DataLoader()
    df = loader.load_base_ids()
    df = loader.load_and_merge_features(df)
    
    # Mocking features for demonstration if loading is skipped/incomplete
    # In production, ensure data is fully loaded
    missing_cols = Config.COLS_SPEND + Config.COLS_COUNT + Config.COLS_BALANCE
    for col in missing_cols:
        if col not in df.columns:
            df[col] = np.random.randint(0, 100000, size=len(df))
            
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.calculate_slopes(df)
    df = engineer.calculate_risk_scores(df)
    df = engineer.assign_churn_segment(df)
    
    # 3. Target Generation
    target_gen = TargetGenerator()
    df = target_gen.generate_target(df)
    
    # 4. Model Training
    # Define features to use (exclude meta columns)
    exclude_cols = [Config.COL_ID, Config.COL_DATE, 'Target', 'Churn_Segment']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    trainer = ModelTrainer()
    model, metrics = trainer.train(df, feature_cols)
    
    # 5. Cleanup
    clean_mem()
    
    elapsed = time.time() - start_time
    print(f"=== Pipeline Completed in {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()
