import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. 찐 데이터 로드 (data 폴더 안에서 불러오기)
df_data = pd.read_csv('data/forty-scan_data.csv')
df_feature = pd.read_csv('data/forty-scan_feature.csv')

# 결측치 및 의미 없는 특징(예: ,,) 제거
features = [str(f).strip() for f in df_feature['feature'].dropna().tolist() if len(str(f).strip()) > 0 and str(f).strip() != ',,']

# 2. Multi-hot 벡터화 함수 정의 (등장 횟수 카운트)
def text_to_multihot(text, feature_list):
    vector = []
    text = str(text)
    for feat in feature_list:
        # 정규식 메타문자 이스케이프 처리
        pattern = re.escape(feat)
        matches = re.findall(pattern, text)
        vector.append(len(matches))
    return np.array(vector)

# 전체 데이터셋에 벡터화 적용
print("데이터 벡터화 진행 중... 🏃‍♂️💨")
X = np.array([text_to_multihot(text, features) for text in df_data['text']])
y = df_data['label'].values

# 3. 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. 모델 학습 (Logistic Regression 베이스라인)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 5. 모델 평가 (4대 지표 확인)
y_pred = model.predict(X_test)
print("\n=== 📊 분류 성능 평가 (Classification Report) ===")
print(classification_report(y_test, y_pred, target_names=['일반 말투(0)', '영포티(1)']))

# 6. 영포티 지수 산출 및 기여 특징 분석 함수
def predict_young_forty(text, model, feature_list):
    vec = text_to_multihot(text, feature_list)
    vec_reshaped = vec.reshape(1, -1)
    
    prob = model.predict_proba(vec_reshaped)[0][1] * 100
    
    contributions = []
    coefficients = model.coef_[0]
    
    for i, count in enumerate(vec):
        if count > 0:
            impact = count * coefficients[i]
            if impact > 0:
                contributions.append((feature_list[i], count, impact))
                
    contributions.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n💬 입력된 대화: '{text}'")
    print(f"📈 영포티 지수: {prob:.2f}%")
    
    if contributions:
        print("🔍 주요 검출 특징 (기여도 순):")
        for feat, count, imp in contributions:
            print(f"   - '{feat}' ({count}회 등장)")
    else:
        print("🔍 검출된 영포티 특징이 없거나, 점수에 영향을 주지 않았습니다 ㅎㅎ")

# -----------------------------------------------------------------
# 7. 🌟 직접 입력해서 테스트하는 사용자 인터랙티브 모드 (신규 추가!)
# -----------------------------------------------------------------
print("\n" + "="*55)
print("😎 영포티 판독기 실시간 측정 모드! (종료하려면 'q' 또는 '종료')")
print("="*55)

while True:
    user_input = input("\n측정할 문장을 입력하세요 >> ").strip()
    
    # 종료 조건 체크
    if user_input.lower() in ['q', 'quit', 'exit', '종료', 'ㅂㅂ']:
        print("\n판독기를 종료합니다. 오늘도 젊고 활기찬 하루 되세요 ^^ 😎")
        break
        
    # 빈 입력값 방지
    if not user_input:
        print("문장을 입력해주세요~ ㅎㅎ")
        continue
    
    # 지수 평가 실행
    predict_young_forty(user_input, model, features)