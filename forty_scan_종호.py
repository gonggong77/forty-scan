import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 0. 한글 폰트 및 글로벌 스타일 설정
# ---------------------------------------------------------
plt.rc("font", family="Malgun Gothic")  # Windows 기준 한글 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
plt.rcParams.update({"font.size": 10})  # 기본 폰트 크기 고정

# ---------------------------------------------------------
# 1. 데이터 로드 (지정된 폴더 내 모든 csv 통합)
# ---------------------------------------------------------
DATA_DIR = r"C:\MyWork\machine_learning\forty_scan\forty-scan\data"
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

df_list = []
for filename in all_files:
    # csv 파일에 'text'와 'label' 컬럼이 포함되어 있다고 가정
    temp_df = pd.read_csv(filename)
    df_list.append(temp_df)

if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    # 테스트용 더미 데이터 생성 (데이터 경로에 파일이 없을 경우 작동)
    df = pd.DataFrame(
        {
            "text": [
                "오늘 회의 몇 시인가요?",
                "오호라~! 우리 OOO 대리님 주말 잘 보냈냥? ^^",
                "프로젝트 일정 공유드립니다.",
                "열정! 열정! 열정! 오늘 회식 고고씽~!!",
            ]
            * 250,
            "label": [0, 1, 0, 1] * 250,
        }
    )

print(f"전체 데이터 개수: {len(df)}")
print(f"클래스 분포:\n{df['label'].value_counts(normalize=True)}")

# ---------------------------------------------------------
# 2. 딥러닝/복잡한 NLP 제외한 텍스트 수치 특징(Feature) 추출
# ---------------------------------------------------------


def extract_text_features(text_series):
    features = pd.DataFrame()

    # 1. 텍스트 길이 및 단어 수
    features["char_count"] = text_series.apply(len)
    features["word_count"] = text_series.apply(lambda x: len(str(x).split()))

    # 2. 특수문자 및 물결(~), 느낌표(!), 물음표(?) 수
    features["exclamation_count"] = text_series.apply(
        lambda x: str(x).count("!") + str(x).count("~")
    )
    features["question_count"] = text_series.apply(lambda x: str(x).count("?"))
    features["special_char_count"] = text_series.apply(
        lambda x: len(re.findall(r"[^가-힣a-zA-A0-9\s]", str(x)))
    )

    # 3. ㅋㅋㅋ, ㅎㅎㅎ 등 자음 반복 및 이모티콘 기호 사용 횟수
    features["korean_cons_count"] = text_series.apply(
        lambda x: len(re.findall(r"[ㄱ-ㅎ]", str(x)))
    )
    features["emoticon_count"] = text_series.apply(
        lambda x: len(re.findall(r"[\^\_\^\;\:\)]", str(x)))
    )

    return features


# 통계적 수치 특징 추출
X_numeric = extract_text_features(df["text"].astype(str))

# 기본 단어 단위 TF-IDF (띄어쓰기 기준 단어 벡터화)
tfidf = TfidfVectorizer(max_features=500, min_df=2)
X_tfidf = tfidf.fit_transform(df["text"].astype(str))

# 수치 데이터와 TF-IDF 행렬 결합
X = hstack([X_numeric.values, X_tfidf])
y = df["label"].values

# ---------------------------------------------------------
# 3. Stratified Train / Test Split (8:2 비율, 5:5 층화)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습 데이터 수: {X_train.shape[0]}, 검증 데이터 수: {X_test.shape[0]}")
print(
    f"Train 클래스 비율 (0:1) = {np.bincount(y_train)[0]} : {np.bincount(y_train)[1]}"
)
print(
    f"Test 클래스 비율 (0:1) = {np.bincount(y_test)[0]} : {np.bincount(y_test)[1]}"
)

# ---------------------------------------------------------
# 4. 모델 학습 (로지스틱 회귀 & LightGBM)
# ---------------------------------------------------------
# 1) 로지스틱 회귀 (Linear Regression 계열 분류기)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# 2) LightGBM 분류기
lgbm_model = LGBMClassifier(
    n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
)
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 5. 성능 측정 및 시각화 (폰트 겹침 방지 및 레이아웃 개선)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# [시각화 1] 로지스틱 회귀 Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)
disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr, display_labels=["정상(0)", "영포티(1)"]
)
disp_lr.plot(ax=axes[0, 0], cmap="Blues", colorbar=False)
axes[0, 0].set_title(
    f"로지스틱 회귀 혼동 행렬\n(F1 Score: {f1_score(y_test, lr_pred):.3f})",
    fontsize=12,
    pad=12,
)
axes[0, 0].set_xlabel("예측 클래스 (Predicted)", fontsize=10.5)
axes[0, 0].set_ylabel("실제 클래스 (Actual)", fontsize=10.5)
axes[0, 0].tick_params(axis="both", labelsize=10)

# [시각화 2] LightGBM Confusion Matrix
cm_lgbm = confusion_matrix(y_test, lgbm_pred)
disp_lgbm = ConfusionMatrixDisplay(
    confusion_matrix=cm_lgbm, display_labels=["정상(0)", "영포티(1)"]
)
disp_lgbm.plot(ax=axes[0, 1], cmap="Greens", colorbar=False)
axes[0, 1].set_title(
    f"LightGBM 혼동 행렬\n(F1 Score: {f1_score(y_test, lgbm_pred):.3f})",
    fontsize=12,
    pad=12,
)
axes[0, 1].set_xlabel("예측 클래스 (Predicted)", fontsize=10.5)
axes[0, 1].set_ylabel("실제 클래스 (Actual)", fontsize=10.5)
axes[0, 1].tick_params(axis="both", labelsize=10)

# [시각화 3] ROC Curve 비교
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgbm_prob)

axes[1, 0].plot(
    fpr_lr,
    tpr_lr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {auc(fpr_lr, tpr_lr):.3f})",
)
axes[1, 0].plot(
    fpr_lgb,
    tpr_lgb,
    linewidth=2,
    linestyle="--",
    label=f"LightGBM (AUC = {auc(fpr_lgb, tpr_lgb):.3f})",
)
axes[1, 0].plot(
    [0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.5)"
)
axes[1, 0].set_xlabel("False Positive Rate (1 - 특이도)", fontsize=10.5)
axes[1, 0].set_ylabel("True Positive Rate (재현율)", fontsize=10.5)
axes[1, 0].set_title("ROC 곡선 성능 비교", fontsize=12, pad=12)
axes[1, 0].tick_params(axis="both", labelsize=10)
axes[1, 0].legend(fontsize=9.5, loc="lower right")
axes[1, 0].grid(True, linestyle=":", alpha=0.6)

# [시각화 4] LightGBM 상위 주요 수치 Feature Importance
feature_names = list(X_numeric.columns) + list(
    tfidf.get_feature_names_out()
)
importances = lgbm_model.feature_importances_
top_idx = np.argsort(importances)[-10:]  # 상위 10개 피처

axes[1, 1].barh(
    range(len(top_idx)),
    importances[top_idx],
    align="center",
    color="skyblue",
)
axes[1, 1].set_yticks(range(len(top_idx)))
axes[1, 1].set_yticklabels([feature_names[i] for i in top_idx], fontsize=9.5)
axes[1, 1].set_xlabel("Feature Importance (중요도)", fontsize=10.5)
axes[1, 1].set_title("LightGBM 상위 10개 주요 판별 피처", fontsize=12, pad=12)
axes[1, 1].tick_params(axis="both", labelsize=10)

# 레이아웃 간격 조정 (여백 확보 및 글씨 겹침 방지)
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.show()

# 상세 평가 리포트 출력
print("\n=== LightGBM 분류 성능 평가 리포트 ===")
print(
    classification_report(
        y_test, lgbm_pred, target_names=["정상대화(0)", "영포티(1)"]
    )
)

# ---------------------------------------------------------
# 6. 사용자 입력 기반 실시간 영포티 판독 함수 및 테스트 루프
# ---------------------------------------------------------
def predict_young_forty(text, model, tfidf_vec):
    # 입력 문장을 Series 변환 후 수치 특징 및 TF-IDF 추출
    s_text = pd.Series([text])
    numeric_feat = extract_text_features(s_text)
    tfidf_feat = tfidf_vec.transform(s_text)

    # 피처 결합
    X_input = hstack([numeric_feat.values, tfidf_feat])

    # 예측 및 영포티(1) 확률 계산
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    return pred, prob


print("\n" + "=" * 50)
print(" [영포티 판독기 실시간 테스트] ")
print("종료하려면 'exit' 또는 'q'를 입력하세요.")
print("=" * 50)

while True:
    user_input = input("\n판독할 대화를 입력하세요: ")
    if user_input.strip().lower() in ["exit", "q"]:
        print("판독기를 종료합니다.")
        break

    if not user_input.strip():
        continue

    pred, prob = predict_young_forty(user_input, lgbm_model, tfidf)

    print("-" * 40)
    if pred == 1:
        print(f"🚨 [영포티 대화 감지!] (영포티 확률: {prob * 100:.1f}%)")
    else:
        print(f"✅ [정상 대화입니다] (영포티 확률: {prob * 100:.1f}%)")
    print("-" * 40)