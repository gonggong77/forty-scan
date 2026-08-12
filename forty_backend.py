# ==========================================================
# forty-scan
# 영포티 말투 판독기 - FastAPI 백엔드
#
# 기능
# 1. 카카오톡 TXT 파일 업로드
# 2. TXT에서 화자 자동 추출
# 3. 사용자가 선택한 화자의 메시지만 분석
# 4. 메시지 1개 = 1개 샘플
# 5. Multi-hot 변환 (오타 보정 포함)
# 6. 저장된 Logistic Regression 모델로 판독
# 7. 문장별 영포티 지수 계산
# 8. 문장별 영포티 지수의 평균 = 최종 영포티 지수
#
# 특징추출(정규식 사전 + 오타 보정)과 등급 기준은
# forty_scan_features.py 를 import 해서 씁니다.
# 학습(forty_scan_model.py) / CLI 판독(forty_scan_test.py) 과
# 완전히 동일한 전처리를 써야 점수가 어긋나지 않습니다.
#
# 실행:
#   python forty_backend.py
#   또는
#   uvicorn forty_backend:app --reload
#
# 필요 패키지:
#   pip install -r requirements.txt
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import re

import joblib
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from forty_scan_features import (
    FEATURE_NAMES,
    MODEL_PATH,
    get_grade,
    text_to_multihot,
)


# ==========================================================
# 1. FastAPI 앱 생성
# ==========================================================

app = FastAPI(
    title="FORTY-SCAN API",
    description="카카오톡 대화 기반 영포티 말투 판독 API",
    version="1.0.0"
)


# ==========================================================
# 2. CORS 설정
#
# 프론트엔드와 백엔드의 주소가 다를 경우 필요합니다.
#
# 개발 단계에서는 일단 전체 허용.
# 실제 배포할 때는 프론트엔드 주소만 허용하는 것을 권장합니다.
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# 3. 모델 로드
#
# 서버가 시작될 때 한 번만 모델을 불러옵니다.
#
# 매 요청마다 .pkl을 다시 읽지 않습니다.
# ==========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\n모델 파일을 찾을 수 없습니다.\n"
        f"모델 경로:\n{MODEL_PATH}\n\n"
        f"forty_scan_model.py 를 먼저 실행해서 모델을 생성하세요.\n"
    )


MODEL = joblib.load(MODEL_PATH)["model"]

# 모델을 다시 학습하지 않은 채 특징 사전만 바꾸면
# 계수가 엉뚱한 차원에 곱해집니다. 시작할 때 바로 잡습니다.
if MODEL.coef_.shape[1] != len(FEATURE_NAMES):

    raise RuntimeError(
        f"\n모델과 특징 사전의 차원이 다릅니다.\n"
        f"모델: {MODEL.coef_.shape[1]}차원 / 특징 사전: {len(FEATURE_NAMES)}차원\n\n"
        f"forty_scan_model.py 를 다시 실행해서 모델을 새로 만드세요.\n"
    )


# 로지스틱 회귀의 절편과 계수를 꺼내 둡니다.
#
# 화자 단위 점수를 낼 때 (섹션 9) 와
# 근거 특징을 정렬할 때 (forty_web.aggregate_features) 씁니다.
INTERCEPT = float(MODEL.intercept_[0])

COEFFICIENTS = MODEL.coef_[0]


# ==========================================================
# 3-1. 화자 단위 집계 상수
#
# 문장별 점수를 그냥 평균내면 두 방향으로 틀립니다.
#
#   표본이 적으면  → 우연히 섞인 영포티 문장 하나가 평균을 지배합니다.
#                    (문장 1개짜리 화자의 7.2% 가 "확진 영포티" 로 나왔습니다)
#
#   표본이 많으면  → 특징이 없는 문장의 점수가 0 이 아니라 23.1 점이라
#                    말이 많은 사람은 무슨 말을 하든 23 쪽으로 끌려갑니다.
#                    진짜 영포티의 검출률이 오히려 떨어집니다.
#
# 그래서 확률을 평균내지 않고, 특징 등장 횟수를 합산해
# "이 사람의 평균적인 문장 한 개" 를 만들어 모델에 한 번만 통과시킵니다.
#
# 주의: 아래 주석의 측정치는 특징 26개 모델(절편 -1.200) 기준입니다.
#       특징 사전을 바꾸거나 모델을 재학습하면 다시 재보아야 합니다.
# ==========================================================

# 축소(shrinkage) 강도.
#
# 평균 낼 때 분모에 이 값을 더합니다. 메시지 k 개를 모아야 절반쯤 믿는다는 뜻입니다.
# 표본이 적은 화자는 자동으로 무증거 기준선(sigmoid(절편) = 23.1 점)쪽으로 당겨집니다.
#
# 실제 데이터 시뮬레이션에서 k=8 이면
# 문장 3개짜리 평범한 화자의 오탐률이 2.6% → 0.0% 로 떨어집니다.
PRIOR_MESSAGE_COUNT = 8

# 집계할 때 문장 하나가 기여할 수 있는 특징 최대 횟수.
#
# 합산 방식은 한 문장 도배에 취약합니다.
# ("안녕하세요~~~~~~~~~~ ^^ ^^ ^^ ^^ ^^" 한 줄이면 물결이 16회)
# 기존 평균 방식은 문장 점수가 100 에서 잘려 이 구멍이 막혀 있었습니다.
#
# 실제 데이터에서 횟수가 3 을 넘는 (문장, 특징) 쌍은 0.16% 뿐이라
# 정상적인 대화는 이 상한에 걸리지 않습니다.
MAX_COUNT_PER_SENTENCE = 3

# 등급을 내보낼 최소 메시지 수.
#
# 미만이면 점수는 그대로 보여주되 등급을 "표본 부족" 으로 바꿉니다.
# 문장 몇 개로 사람을 영포티로 단정할 근거는 없습니다.
MIN_MESSAGES_FOR_GRADE = 5


# ==========================================================
# 4. TXT 디코딩
# ==========================================================

def decode_txt(file_bytes):
    """
    업로드된 카카오톡 TXT의 인코딩을 자동으로 확인합니다.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp949"
    ]

    for encoding in encodings:

        try:

            return file_bytes.decode(
                encoding
            )

        except UnicodeDecodeError:

            continue

    raise HTTPException(
        status_code=400,
        detail="TXT 파일의 인코딩을 확인할 수 없습니다."
    )


# ==========================================================
# 5. 카카오톡 TXT 파싱
# ==========================================================

def parse_kakao_txt(text):
    """
    카카오톡 TXT에서 실제 메시지만 추출합니다.

    인식하는 형태:

    [한병준] [오후 3:20] 아
    [한병준] [오후 3:20] 쌩노가다해서

    날짜 / 저장한 날짜 / 기타 정보는 자동으로 제외합니다.

    반환:

    [
        {
            "speaker": "한병준",
            "message": "아"
        },
        {
            "speaker": "한병준",
            "message": "쌩노가다해서"
        }
    ]
    """

    message_pattern = re.compile(
        r"^\[(?P<speaker>.*?)\]\s+"
        r"\[(?P<time>.*?)\]\s+"
        r"(?P<message>.*)$"
    )

    messages = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        match = message_pattern.match(line)

        if not match:
            # 날짜 / 저장한 날짜 / 기타 정보
            continue

        speaker = match.group(
            "speaker"
        ).strip()

        message = match.group(
            "message"
        ).strip()

        if not message:
            continue

        messages.append(
            {
                "speaker": speaker,
                "message": message
            }
        )

    return messages


# ==========================================================
# 6. 화자 목록 추출
# ==========================================================

def get_speakers(messages):
    """
    카카오톡 대화에 등장한 화자를
    등장 순서대로 중복 없이 반환합니다.
    """

    speakers = []

    for item in messages:

        speaker = item["speaker"]

        if speaker not in speakers:

            speakers.append(
                speaker
            )

    return speakers


# ==========================================================
# 7. 업로드 파일 → 메시지 목록
#
# /speakers 와 /analyze 가 똑같이 거치는 앞단 과정입니다.
# 두 곳에 같은 코드를 두면 검증 규칙이 갈라지므로 한 곳으로 모읍니다.
# ==========================================================

async def load_messages(file):
    """
    업로드 파일을 검증하고 카카오톡 메시지 목록으로 만듭니다.

    확장자 확인 → 읽기 → 인코딩 판별 → 파싱 → 빈 결과 확인
    """

    # filename은 None일 수 있으므로 그대로 .lower()를 부르지 않습니다.
    filename = file.filename or ""

    if not filename.lower().endswith(".txt"):

        raise HTTPException(
            status_code=400,
            detail="카카오톡 TXT 파일만 업로드할 수 있습니다."
        )

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    text = decode_txt(
        file_bytes
    )

    messages = parse_kakao_txt(
        text
    )

    if not messages:

        raise HTTPException(
            status_code=400,
            detail="카카오톡 메시지를 찾을 수 없습니다."
        )

    return messages


# ==========================================================
# 8. 문장 하나 분석
# ==========================================================

def analyze_sentence(text):
    """
    문장 하나를 분석합니다.

    반환:

    score
    grade
    detected_features
    vector
    """

    # 학습 때와 동일한 전처리(정규식 + 오타 보정)를 씁니다.
    # fuzzy_flags = 정규식은 놓쳤는데 오타 보정이 잡아낸 특징 이름들
    vector, fuzzy_flags = text_to_multihot(
        text,
        return_fuzzy_flags=True
    )

    # Logistic Regression
    # label=1(영포티)일 확률
    probability = MODEL.predict_proba(
        vector.reshape(1, -1)
    )[0][1]

    score = float(
        probability * 100
    )

    grade, comment = get_grade(
        score
    )

    # 검출된 특징
    detected_features = []

    for name, count in zip(
        FEATURE_NAMES,
        vector
    ):

        if count > 0:

            detected_features.append(
                {
                    "name": name,
                    "count": int(count),
                    "is_fuzzy": name in fuzzy_flags
                }
            )

    return {
        "text": text,
        "score": round(score, 2),
        "grade": grade,
        "grade_comment": comment,
        "detected_features": detected_features,
        "vector": vector.astype(int).tolist()
    }


# ==========================================================
# 9. 최종 대화 분석
# ==========================================================

def analyze_conversation(
    selected_messages
):
    """
    선택된 사람의 전체 메시지를 분석합니다.

    핵심:

    문장별 특징 횟수
          ↓
    전부 합산 후 (메시지 수 + PRIOR_MESSAGE_COUNT) 로 나눔
          ↓
    "이 사람의 평균적인 문장 한 개"
          ↓
    모델에 한 번 통과
          ↓
    최종 영포티 지수

    왜 문장별 점수의 평균이 아닌가:

    문장 단위 확률은 0 / 100 양끝에 몰려 있습니다.
    ("넵~^^" 한 줄이 99.8 점)

    그래서 평균을 내면 표본 수에 따라 의미가 달라집니다.

        문장 1개  → 그 한 문장이 곧 판정 (오탐)
        문장 100개 → 특징 없는 문장(23.1점)에 희석되어 아무도 영포티가 안 됨

    반면 특징 횟수는 더할 수 있는 양이라 표본이 늘수록 추정이 정확해집니다.
    모델이 횟수 벡터에 대한 로지스틱 회귀이므로

        절편 + 계수 · (평균 횟수 벡터)

    는 "이 사람의 평균적 문장" 의 정직한 점수이고,
    학습 때와 같은 모델을 그대로 씁니다.

    분모의 PRIOR_MESSAGE_COUNT 가 표본 적은 화자를 기준선으로 당깁니다.
    """

    if not selected_messages:

        raise HTTPException(
            status_code=400,
            detail="분석할 메시지가 없습니다."
        )

    sentence_results = []

    for message in selected_messages:

        result = analyze_sentence(
            message
        )

        sentence_results.append(
            result
        )

    # ------------------------------------------------------
    # 화자 전체의 특징 횟수 합산
    #
    # analyze_sentence 가 이미 계산해 둔 vector 를 재사용하므로
    # 추가 비용은 덧셈뿐입니다. 모델을 다시 부르지 않습니다.
    #
    # 문장당 MAX_COUNT_PER_SENTENCE 로 자릅니다.
    # 물결 도배 한 줄이 화자 점수를 지배하는 것을 막습니다.
    # ------------------------------------------------------

    total_counts = np.zeros(
        len(COEFFICIENTS)
    )

    for result in sentence_results:

        total_counts += np.minimum(
            np.array(result["vector"], dtype=float),
            MAX_COUNT_PER_SENTENCE
        )

    message_count = len(
        selected_messages
    )

    # ------------------------------------------------------
    # 축소된 평균 → 최종 영포티 지수
    #
    # 분모가 message_count 가 아니라 message_count + PRIOR_MESSAGE_COUNT 입니다.
    # 표본이 적으면 분모가 부풀어 무증거 기준선(23.1점)쪽으로 당겨집니다.
    # ------------------------------------------------------

    average_counts = total_counts / (
        message_count + PRIOR_MESSAGE_COUNT
    )

    logit = INTERCEPT + float(
        average_counts @ COEFFICIENTS
    )

    final_score = float(
        100.0 / (1.0 + np.exp(-logit))
    )

    # ------------------------------------------------------
    # 등급
    #
    # 표본이 모자라면 점수는 그대로 보여주되 등급은 보류합니다.
    # 축소 덕분에 점수 자체는 이미 기준선 근처라 안전하지만,
    # "정상 범주" 라고 단정하는 것도 근거가 없기는 마찬가지입니다.
    # ------------------------------------------------------

    is_provisional = message_count < MIN_MESSAGES_FOR_GRADE

    if is_provisional:

        final_grade = "표본 부족"

        final_comment = (
            f"문장이 {message_count}개뿐이라 판정을 보류합니다."
        )

    else:

        final_grade, final_comment = get_grade(
            final_score
        )

    # 이 점수를 얼마나 믿을 수 있는지. 0(근거 없음) ~ 1(충분)
    reliability = message_count / (
        message_count + PRIOR_MESSAGE_COUNT
    )

    return {
        "final_score": round(
            final_score,
            2
        ),
        "final_grade": final_grade,
        "final_comment": final_comment,
        "message_count": message_count,
        "reliability": round(
            reliability,
            2
        ),
        "is_provisional": is_provisional,
        "sentence_results": sentence_results
    }


# ==========================================================
# 9-1. 문장 판독 요청 바디
#
# forty_scan_test.py 의 대화형 입력(문장 1개)을
# API로 그대로 옮기기 위한 요청 스키마입니다.
# ==========================================================

class SentenceRequest(BaseModel):
    text: str


# ==========================================================
# 10. 서버 상태 확인
# ==========================================================

@app.get("/")
def root():

    return {
        "service": "FORTY-SCAN",
        "status": "running",
        "model": "LogisticRegression"
    }


# ==========================================================
# 11. 화자 목록 확인 API
# ==========================================================

@app.post("/speakers")
async def get_speaker_list(
    file: UploadFile = File(...)
):
    """
    프론트엔드에서 TXT 파일을 업로드하면
    해당 파일에 존재하는 화자 목록을 반환합니다.

    프론트엔드는 이 API를 먼저 호출해서

    [
        "한병준",
        "16이도현"
    ]

    같은 목록을 얻은 후
    사용자에게 선택하게 만들면 됩니다.
    """

    messages = await load_messages(
        file
    )

    speakers = get_speakers(
        messages
    )

    return {
        "filename": file.filename,
        "speakers": speakers,
        "message_count": len(messages)
    }


# ==========================================================
# 12. 실제 분석 API
# ==========================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    speaker: str = Form(...)
):
    """
    카카오톡 TXT + 선택한 화자를 받아
    실제 영포티 분석을 수행합니다.

    요청:

        file    = 카카오톡.txt
        speaker = 한병준

    반환:

        final_score
        final_grade
        sentence_results
        ...
    """

    # ------------------------------------------------------
    # 1. 업로드 파일 → 카카오톡 메시지 목록
    # ------------------------------------------------------

    messages = await load_messages(
        file
    )

    # ------------------------------------------------------
    # 2. 화자 목록 확인
    # ------------------------------------------------------

    speakers = get_speakers(
        messages
    )

    # ------------------------------------------------------
    # 3. 존재하지 않는 화자 방지
    # ------------------------------------------------------

    if speaker not in speakers:

        raise HTTPException(
            status_code=400,
            detail={
                "message": "선택한 화자를 찾을 수 없습니다.",
                "selected_speaker": speaker,
                "available_speakers": speakers
            }
        )

    # ------------------------------------------------------
    # 4. 선택된 화자의 메시지만 추출
    # ------------------------------------------------------

    selected_messages = [
        item["message"]
        for item in messages
        if item["speaker"] == speaker
    ]

    if not selected_messages:

        raise HTTPException(
            status_code=400,
            detail="선택한 화자의 메시지가 없습니다."
        )

    # ------------------------------------------------------
    # 5. 분석
    # ------------------------------------------------------

    result = analyze_conversation(
        selected_messages
    )

    # ------------------------------------------------------
    # 6. 최종 결과 반환
    # ------------------------------------------------------

    return {
        "filename": file.filename,
        "speaker": speaker,

        # 전체 대화의 핵심 결과
        "final_score": result["final_score"],
        "final_grade": result["final_grade"],
        "final_comment": result["final_comment"],

        # 분석 문장 수
        "message_count": result["message_count"],

        # 표본이 얼마나 되는지
        #
        # reliability     : 0(근거 없음) ~ 1(충분)
        # is_provisional  : True 면 등급이 "표본 부족" 이고 판정 보류
        "reliability": result["reliability"],
        "is_provisional": result["is_provisional"],

        # 참고용
        "available_speakers": speakers,

        # 문장별 상세 결과
        "sentence_results": result["sentence_results"]
    }


# ==========================================================
# 13. 문장 하나 판독 API
#
# forty_scan_test.py 의 대화형 판독 기능을 그대로 옮긴 API입니다.
# TXT 업로드 없이, 문장 하나만 보내면 바로 판독 결과를 돌려줍니다.
#
# 요청 (JSON):
#
#     { "text": "야 이거 레알 오지구요 지리구요" }
#
# 반환:
#
#     analyze_sentence() 의 결과와 동일
#     (score, grade, grade_comment, detected_features, vector)
# ==========================================================

@app.post("/analyze-sentence")
def analyze_single_sentence(
    request: SentenceRequest
):

    text = request.text.strip()

    if not text:

        raise HTTPException(
            status_code=400,
            detail="분석할 문장이 없습니다."
        )

    # 8번 섹션에 이미 정의된 함수를 그대로 재사용합니다.
    # /analyze 가 문장마다 호출하는 것과 완전히 같은 로직입니다.
    result = analyze_sentence(
        text
    )

    return result


# ==========================================================
# 실행 방법
#
# 터미널:
#
# python forty_backend.py
#
# 또는:
#
# uvicorn forty_backend:app --reload
#
# 이후:
#
# http://127.0.0.1:8000/docs
#
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "forty_backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )