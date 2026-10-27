# Historical source excerpts — DO NOT RUN as a standalone script
# No source logic has been repaired. Training calls are archival, not instructions.
# Source: LG_Aimers_6트.ipynb; original SHA256: 9eb053c8918417cce43727e55521dbbb664de725cf350004138d1c057e3c7f97
# Cell numbers include markdown cells and are 1-based; execution counters are separate.

# ===== Original cell 5; execution_count=9 =====
train_df['특정 시술 유형'] = train_df['특정 시술 유형'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
test_df['특정 시술 유형'] = train_df['특정 시술 유형'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

# ===== Original cell 6; execution_count=11 =====
def feature_engineering_fix(df):
    df_new = df.copy()
    epsilon = 1e-6

    embryo_cols = ['총 생성 배아 수', '미세주입된 난자 수', '미세주입에서 생성된 배아 수',
                   '수집된 신선 난자 수', '혼합된 난자 수', '파트너 정자와 혼합된 난자 수', '저장된 배아 수']
    for col in embryo_cols:
        df_new[col] = df_new[col].fillna(0)
    
    df_new['ICSI 효율'] = np.where(df_new['미세주입된 난자 수'] > 0,
                                  df_new['미세주입에서 생성된 배아 수'] / (df_new['미세주입된 난자 수'] + epsilon),
                                  0)
    df_new['신선 난자 활용률'] = np.where(df_new['수집된 신선 난자 수'] > 0,
                                      df_new['총 생성 배아 수'] / (df_new['수집된 신선 난자 수'] + epsilon),
                                      0)
    df_new['파트너 정자 혼합 비율'] = np.where(df_new['혼합된 난자 수'] > 0,
                                             df_new['파트너 정자와 혼합된 난자 수'] / (df_new['혼합된 난자 수'] + epsilon),
                                             0)
    df_new['배아 저장 비율'] = np.where(df_new['총 생성 배아 수'] > 0,
                                      df_new['저장된 배아 수'] / (df_new['총 생성 배아 수'] + epsilon),
                                      0)
    df_new['총 난자 수'] = (df_new['미세주입된 난자 수'] +
                           df_new['수집된 신선 난자 수'] +
                           df_new['혼합된 난자 수'])
    df_new['전체 배아 생성률'] = np.where(df_new['총 난자 수'] > 0,
                                         df_new['총 생성 배아 수'] / (df_new['총 난자 수'] + epsilon),
                                         0)
    

    df_new['생성-저장 배아 차이'] = df_new['총 생성 배아 수'] - df_new['저장된 배아 수']
    df_new['미세주입 난자 비율'] = np.where(df_new['총 난자 수'] > 0,
                                          df_new['미세주입된 난자 수'] / (df_new['총 난자 수'] + epsilon),
                                          0)
    df_new['신선 난자 비율'] = np.where(df_new['총 난자 수'] > 0,
                                      df_new['수집된 신선 난자 수'] / (df_new['총 난자 수'] + epsilon),
                                      0)
    df_new['혼합 난자 비율'] = np.where(df_new['총 난자 수'] > 0,
                                      df_new['혼합된 난자 수'] / (df_new['총 난자 수'] + epsilon),
                                      0)
        
    df_new['ICSI 효율 x 총 난자 수'] = df_new['ICSI 효율'] * df_new['총 난자 수']
    df_new['신선 난자 활용률 x 총 난자 수'] = df_new['신선 난자 활용률'] * df_new['총 난자 수']
    df_new['ICSI 효율 x 신선 난자 활용률'] = df_new['ICSI 효율'] * df_new['신선 난자 활용률']
    df_new['배아 저장 비율 x 전체 배아 생성률'] = df_new['배아 저장 비율'] * df_new['전체 배아 생성률']
    df_new['배아 저장 비율 x 전체 배아 생성률_0'] = np.where(df_new['배아 저장 비율 x 전체 배아 생성률'] == 0,0,1)
    df_new['ICSI 효율 / 신선 난자 활용률'] = np.where(df_new['신선 난자 활용률'] > 0,
                                                      df_new['ICSI 효율'] / (df_new['신선 난자 활용률'] + epsilon),
                                                      0)
    df_new['배아 저장 비율 / 전체 배아 생성률'] = np.where(df_new['전체 배아 생성률'] > 0,
                                                         df_new['배아 저장 비율'] / (df_new['전체 배아 생성률'] + epsilon),
                                                         0)

    df_new['임신 시도 여부'] = df_new['임신 시도 또는 마지막 임신 경과 연수'].notnull().astype(int)
    df_new['임신 시도 또는 마지막 임신 경과 연수'] = df_new['임신 시도 또는 마지막 임신 경과 연수'].fillna(0)
    

    
    cause_cols1 = ["남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", 
                   "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인"]
    df_new['부부 불임 원인 합계'] = df_new[cause_cols1].sum(axis=1)
    
    cause_cols2 = ["불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", 
                   "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
                   "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성",
                   "불임 원인 - 정자 형태"]
    df_new['불임 원인 세부 합계'] = df_new[cause_cols2].sum(axis=1)
    
    # catboost 고점용 추가 파생변수
    df_new['동결 배아 사용률'] = df_new['해동된 배아 수'] / (df_new['저장된 배아 수'] + epsilon)
                                 
    df_new['배아 생존율'] = df_new['이식된 배아 수'] / (df_new['해동된 배아 수'] + epsilon)
    
    df_new['이식된 배아 수 / 저장된 배아 수'] = df_new['이식된 배아 수'] / (df_new['저장된 배아 수'] + epsilon)

    
    return df_new

# ===== Original cell 7; execution_count=13 =====
train_df = feature_engineering_fix(train_df)
test_df = feature_engineering_fix(test_df)

# ===== Original cell 8; execution_count=15 =====
def num_to_cat(df, cols):
    df_new = df.copy()
    for col in cols:
        df_new[col + '_num'] = pd.to_numeric(df_new[col], errors='coerce')
        df_new[col + '_cate'] = df_new[col + '_num'].astype('category')
        df_new = df_new.drop(columns = [col])
    return df_new

cols_to_convert = ['이식된 배아 수', '미세주입 배아 이식 수', '배아 이식 경과일','저장된 배아 수']

# ===== Original cell 9; execution_count=17 =====
train_df = num_to_cat(train_df, cols_to_convert)
test_df = num_to_cat(test_df, cols_to_convert)

# ===== Original cell 10; execution_count=19 =====
def engineer_additional_features(df):
    
    df_new = df.copy()
    epsilon = 1e-6  
    import re
    def convert_age(age_str):

        if isinstance(age_str, str):
            match = re.search(r'만(\d+)', age_str)
            if match:
                return int(match.group(1))
        return np.nan

    df_new['시술 당시 나이_숫자'] = df_new['시술 당시 나이'].apply(convert_age)
    df_new['이식된/총생성비'] = df_new['이식된 배아 수_num'] / (df_new['총 생성 배아 수'] + epsilon)
    df_new['저장된/이식된비'] = df_new['저장된 배아 수_num'] / (df_new['이식된 배아 수_num'] + epsilon)
    df_new['이식일수/나이'] = df_new['배아 이식 경과일_num'] / (df_new['시술 당시 나이_숫자'] + epsilon)
    df_new['이식수x이식일수'] = df_new['이식된 배아 수_num'] * df_new['배아 이식 경과일_num']
    df_new['저장-이식 차이'] = df_new['저장된 배아 수_num'] - df_new['이식된 배아 수_num']

    if '난자 채취 경과일' in df_new.columns:
        df_new['이식일수/난자채취일'] = df_new['배아 이식 경과일_num'] / (df_new['난자 채취 경과일'] + epsilon)
    
    df_new['배아 저장 비율 x 전체 배아 생성률_0'] = pd.to_numeric(df_new['배아 저장 비율 x 전체 배아 생성률_0'], errors='coerce')
    df_new['저장비율_전체생성_상호작용'] = df_new['배아 저장 비율 x 전체 배아 생성률_0'] * df_new['저장된 배아 수_num']
    
    return df_new

# ===== Original cell 11; execution_count=21 =====
train_df = engineer_additional_features(train_df)
test_df = engineer_additional_features(test_df)

# ===== Original cell 12; execution_count=23 =====
import numpy as np
import pandas as pd

def make_more_feature(df):
    df_new = df.copy()
    epsilon = 1e-6
    
    df_new['하루 평균 이식 배아 수'] = df_new['이식된 배아 수_num'] / (df_new['배아 이식 경과일_num'] + epsilon)
    df_new['이식 수와 경과일 상호작용'] = df_new['이식된 배아 수_num'] * df_new['배아 이식 경과일_num']
    df_new['연령 기준 이식 배아 수'] = df_new['이식된 배아 수_num'] / (df_new['시술 당시 나이_숫자'] + epsilon)
    df_new['연령 기준 배아 이식 경과일'] = df_new['배아 이식 경과일_num'] / (df_new['시술 당시 나이_숫자'] + epsilon)
    df_new['경과일과 이식 배아 수 차이'] = df_new['배아 이식 경과일_num'] - df_new['이식된 배아 수_num']
    df_new['연령과 이식 배아 수 상호작용'] = df_new['시술 당시 나이_숫자'] * df_new['이식된 배아 수_num']
    df_new['연령과 배아 이식 경과일 상호작용'] = df_new['시술 당시 나이_숫자'] * df_new['배아 이식 경과일_num']
    df_new['이식 범주 결합'] = df_new['이식된 배아 수_cate'].astype(str) + '_' + df_new['배아 이식 경과일_cate'].astype(str)
    
    return df_new

# ===== Original cell 13; execution_count=25 =====
train_df = make_more_feature(train_df)
test_df = make_more_feature(test_df)

# ===== Original cell 14; execution_count=27 =====
def more_feature2(df):
    df_new = df.copy()
    epsilon = 1e-6
    
    def cat2num(count):
        mapping = {str(i) + '회': i for i in range(6)}
        mapping['6회 이상'] = 6
        
        return mapping[count]

    cols = ['총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수']
    for col in cols:
        df_new[col + '_num'] = df_new[col].apply(cat2num)

    # 시술 대비 임신 횟수
    df_new['총 시술 대비 임신 횟수'] = np.where(df_new['총 시술 횟수_num'] > 0, df_new['총 임신 횟수_num']/df_new['총 시술 횟수_num'], 0)
    df_new['IVF 시술 대비 임신 횟수'] = np.where(df_new['IVF 시술 횟수_num'] > 0, df_new['IVF 임신 횟수_num']/df_new['IVF 시술 횟수_num'], 0)
    df_new['DI 시술 대비 임신 횟수'] = np.where(df_new['DI 시술 횟수_num'] > 0, df_new['DI 임신 횟수_num']/df_new['DI 시술 횟수_num'], 0)

    # 임신 대비 출산 횟수
    df_new['총 임신 대비 출산 횟수'] = np.where(df_new['총 임신 횟수_num'] > 0, df_new['총 출산 횟수_num']/df_new['총 임신 횟수_num'], 0)
    df_new['IVF 임신 대비 출산 횟수'] = np.where(df_new['IVF 임신 횟수_num'] > 0, df_new['IVF 출산 횟수_num']/df_new['IVF 임신 횟수_num'], 0)
    df_new['DI 임신 대비 출산 횟수'] = np.where(df_new['DI 임신 횟수_num'] > 0, df_new['DI 출산 횟수_num']/df_new['DI 임신 횟수_num'], 0)


    ## 시술 당시 나이 관련
    

    return df_new

# ===== Original cell 15; execution_count=29 =====
train_df = more_feature2(train_df)
test_df = more_feature2(test_df)

# ===== Original cell 16; execution_count=31 =====
categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부",
    "미세주입 배아 이식 수_cate",
    "배아 이식 경과일_cate",
    "이식된 배아 수_cate" ,'저장된 배아 수_cate',
    '배아 저장 비율 x 전체 배아 생성률_0', 
    "이식 범주 결합"
]

# ===== Original cell 17; execution_count=33 =====
from tqdm import tqdm
train_encoded = train_df.copy()
test_encoded = test_df.copy()

for feature in tqdm(categorical_columns):
    le = LabelEncoder()
    train_encoded[feature] = train_encoded[feature].astype(str) 
    test_encoded[feature] = test_encoded[feature].astype(str) 
    le.fit(train_encoded[feature]) 
    test_encoded[feature] = test_encoded[feature].apply(lambda x: x if x in le.classes_ else "unknown")

    le.classes_ = np.append(le.classes_, "unknown")  
    train_encoded[feature] = le.transform(train_encoded[feature])
    test_encoded[feature] = le.transform(test_encoded[feature])

# ===== Original cell 18; execution_count=35 =====
X_encoded = train_encoded.drop('임신 성공 여부', axis = 1)
y = train_df['임신 성공 여부']

# ===== Original cell 20; execution_count=18 =====
# 최적화 해서 나온 파라미터
best_params = {'iterations': 1400,
 'od_wait': 1211,
 'learning_rate': 0.04327758581699257,
 'reg_lambda': 49.09396260836614,
 'random_strength': 41.89925594614459,
 'depth': 9,
 'min_data_in_leaf': 28,
 'leaf_estimation_iterations': 1,
 'bagging_temperature': 0.022029436061740598,
 'colsample_bylevel': 0.5405765517905377,
 'scale_pos_weight': 3,
              'eval_metric':'AUC',
              'verbose' : 100}

# ===== Original cell 21; execution_count=19 =====
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

FEATURES = [col for col in X_encoded if col not in ['임신 성공 여부']]
cat_features = categorical_columns

cb_params = {
 'eval_metric' : 'AUC',
 'verbose' : 100}

cb_predictions = []
cb_fimp = []
cb_score = []

kf = KFold(n_splits=5, shuffle=True, random_state=21)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_encoded, y)):
    print(10 * "=", f"Fold={fold+1}", 10 * "=")

    X_train, X_valid = X_encoded.iloc[train_idx], X_encoded.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    eval_pool = Pool(X_valid, y_valid, cat_features=cat_features)
    test_pool = Pool(data = test_encoded, cat_features=cat_features)

    model_cat = CatBoostClassifier(**best_params)
    model_cat.fit(train_pool, eval_set=eval_pool, use_best_model=True, early_stopping_rounds=50)

    preds_valid = model_cat.predict_proba(eval_pool)[:, 1]  
    auc = roc_auc_score(y_valid, preds_valid) 
    cb_score.append(auc)

    print(f"Fold={fold+1}, AUC: {auc:.4f}")

    fim = pd.DataFrame(index=FEATURES, data=model_cat.feature_importances_)
    cb_fimp.append(fim)
    cat_test_preds = model_cat.predict_proba(test_pool)[:, 1]  
    cb_predictions.append(cat_test_preds)

print("Mean AUC:", np.mean(cb_score))

# ===== Original cell 22; execution_count=27 =====
pd.set_option('display.max_rows', 30)

df_concat = pd.concat(cb_fimp, axis=1)
df_mean = df_concat.mean(axis=1)
df_result = pd.DataFrame(df_mean, columns=['평균 중요도'])
df_result.sort_values(by='평균 중요도', ascending=False).head(30)

# ===== Original cell 26; execution_count=21 =====
pd.set_option('display.max_rows', 30)

df_concat = pd.concat(cb_fimp, axis=1)
df_mean = df_concat.mean(axis=1)
df_result = pd.DataFrame(df_mean, columns=['평균 중요도'])
df_result.sort_values(by='평균 중요도', ascending=False).head(30)

# ===== Original cell 32; execution_count=39 =====
best_params = {'num_leaves': 14,
 'max_depth': 5,
 'learning_rate': 0.047478677958161105,
 'n_estimators': 450,
 'class_weight': 'balanced',
 'min_child_samples': 50,
 'subsample': 0.9522278487928603,
 'colsample_bytree': 0.8084155876319679,
 'reg_alpha': 0.882902124107087,
 'reg_lambda': 8.479301899218871,
 'metric': 'auc', "objective" : 'binary'}

# ===== Original cell 33; execution_count=37 =====
X_encoded[categorical_columns] = X_encoded[categorical_columns].astype('float')
test_encoded[categorical_columns] = test_encoded[categorical_columns].astype('float')

# ===== Original cell 35; execution_count=49 =====
## LightGBM
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

FEATURES = [col for col in X_encoded if col not in ['임신 성공 여부']]

rb_params = {
    "verbose": 0,
    "objective" : 'binary',
    "metric" : 'auc',
    'learning_rate': 0.047478677958161105
}

rb_predictions = []
rb_fimp = []
rb_score = []

kf = KFold(n_splits=5, shuffle=True, random_state=21)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_encoded, y)):
    print(10 * "=", f"Fold={fold+1}", 10 * "=")
    
    X_train, X_valid = X_encoded.iloc[train_idx], X_encoded.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    eval_set = [(X_valid, y_valid)]
    
    model_rb = lgb.LGBMClassifier(**best_params)
    model_rb.fit(X_train, y_train, eval_metric='auc', eval_set=eval_set, categorical_feature=cat_features)

    preds_valid = model_rb.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds_valid)
    rb_score.append(auc)

    print(f"Fold={fold+1}, AUC: {auc:.4f}")
    
    fim = pd.DataFrame(index=FEATURES,
                       data=model_rb.feature_importances_,
                       columns=[f'Fold_{fold+1}'])
    rb_fimp.append(fim)
    
    rb_test_preds = model_rb.predict_proba(test_encoded)[:, 1] 
    rb_predictions.append(rb_test_preds)

print("Mean AUC:", np.mean(rb_score))

# ===== Original cell 39; execution_count=101 =====
pd.set_option('display.max_rows', 40)

df_concat = pd.concat(rb_fimp, axis=1)
df_mean = df_concat.mean(axis=1)
df_result = pd.DataFrame(df_mean, columns=['평균 중요도'])
df_result.sort_values(by='평균 중요도', ascending=False).head(40)

# ===== Original cell 46; execution_count=284 =====
ensemble_result = (cb_result + rb_result) / 2
ensemble_result

# ===== Original cell 49; execution_count=287 =====
sub_df['probability'] = (cb_result + auto_['probability'] + rb_result) / 3
sub_df

# ===== Original cell 53; execution_count=53 =====
from autogluon.tabular import TabularDataset, TabularPredictor

for col in categorical_columns:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)

label  = "임신 성공 여부"
metric = "roc_auc"

time_limit = 3600
num_trials = 5  
search_strategy = 'auto'

hyperparameter_tune_kwargs = {  
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
    'random_state' : 21
}

predictor = TabularPredictor(label=label, eval_metric=metric).fit(train_data,
    time_limit = time_limit,
     hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    excluded_model_types = ['KNN', 'NN', 'FASTAI', 'NN_TORCH']
)

# excluded_model_types = ['KNN', 'NN_TORCH']

# ===== Original cell 54; execution_count=222 =====
leader_temp = predictor.leaderboard(train_data, silent=True)

# ===== Original cell 55; execution_count=223 =====
leader_temp.sort_values(by='score_val', ascending=False)

# ===== Original cell 56; execution_count=210 =====
predictor.feature_importance(data = train_data).head(20)

# ===== Original cell 71; execution_count=109 =====
tmp = (bo['probability'] + auto['probability'] + sub_df['probability']) / 3 
tmp
