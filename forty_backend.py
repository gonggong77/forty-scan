
# ==========================================================
# forty-scan
# 영포티 말투 판독기 - FastAPI 백엔드
#
# 기능
# 1. 카카오톡 TXT 파일 업로드
# 2. TXT에서 화자 자동 추출
# 3. 사용자가 선택한 화자의 메시지만 분석
# 4. 메시지 1개 = 1개 샘플
# 5. 기존 28개 특징 Multi-hot 변환
# 6. 저장된 Logistic Regression 모델로 판독
# 7. 문장별 영포티 지수 계산
# 8. 문장별 영포티 지수의 평균 = 최종 영포티 지수
#
# 실행:
#   uvicorn main:app --reload
#
# 필요 패키지:
#   pip install fastapi uvicorn python-multipart joblib numpy scikit-learn
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


# ==========================================================
# 1. 기본 설정
# ==========================================================

# ==========================================================
# 1. 기본 설정
# ==========================================================

MODEL_PATH = os.path.join(
    "model",
    "forty_scan_현경_model.pkl"
)

# ==========================================================
# 2. FastAPI 앱 생성
# ==========================================================

app = FastAPI(
    title="FORTY-SCAN API",
    description="카카오톡 대화 기반 영포티 말투 판독 API",
    version="1.0.0"
)


# ==========================================================
# 3. CORS 설정
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
# 4. 모델 로드
#
# 서버가 시작될 때 한 번만 모델을 불러옵니다.
#
# 매 요청마다 .pkl을 다시 읽지 않습니다.
# ==========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\n모델 파일을 찾을 수 없습니다.\n"
        f"모델 경로:\n{MODEL_PATH}\n"
    )


MODEL_DATA = joblib.load(MODEL_PATH)

MODEL = MODEL_DATA["model"]
FEATURES = MODEL_DATA["features"]

FEATURE_NAMES = [
    name
    for name, _ in FEATURES
]

COMPILED_PATTERNS = [
    re.compile(pattern)
    for _, pattern in FEATURES
]


# ==========================================================
# 5. 등급 기준
# ==========================================================

GRADES = [
    (
        20,
        "진성 MZ",
        "영포티 특징이 거의 없습니다."
    ),
    (
        40,
        "정상 범주",
        "무난한 말투입니다."
    ),
    (
        60,
        "영포티 경계",
        "슬슬 물결(~)과 마침표를 조심하세요."
    ),
    (
        80,
        "MZ 코스프레 40대",
        "젊은 척이 살짝 티납니다."
    ),
    (
        101,
        "확진 영포티",
        "라떼 향이 진하게 감지되었습니다."
    )
]


def get_grade(score):
    """
    영포티 지수(0~100)를 5단계 등급으로 변환합니다.
    """

    for upper, name, comment in GRADES:

        if score < upper:

            return {
                "name": name,
                "comment": comment
            }

    return {
        "name": GRADES[-1][1],
        "comment": GRADES[-1][2]
    }


# ==========================================================
# 6. TXT 디코딩
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
# 7. 카카오톡 TXT 파싱
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
# 8. 화자 목록 추출
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
# 9. Multi-hot 변환
# ==========================================================

def text_to_multihot(text):
    """
    문장 하나를 28차원 Multi-hot 벡터로 변환합니다.

    주의:
    단순 0/1이 아니라
    특징의 등장 횟수를 그대로 사용합니다.

    예:

    "ㅋㅋㅋㅋ 라떼는 말이야!!"

    각 특징의 검출 횟수가
    [0, 0, 1, 2, ...]
    형태로 들어갑니다.
    """

    if text is None:

        text = ""

    text = str(text)

    vector = []

    for pattern in COMPILED_PATTERNS:

        count = len(
            pattern.findall(text)
        )

        vector.append(
            count
        )

    return np.array(
        vector,
        dtype=float
    )


# ==========================================================
# 10. 여러 문장 Multi-hot 변환
# ==========================================================

def transform_texts(texts):
    """
    여러 문장을 한꺼번에 벡터화합니다.

    반환 형태:

    (문장 개수, 28)
    """

    return np.array(
        [
            text_to_multihot(text)
            for text in texts
        ],
        dtype=float
    )


# ==========================================================
# 11. 문장 하나 분석
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

    vector = text_to_multihot(
        text
    )

    # Logistic Regression
    # label=1(영포티)일 확률
    probability = MODEL.predict_proba(
        vector.reshape(1, -1)
    )[0][1]

    score = float(
        probability * 100
    )

    grade = get_grade(
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
                    "count": int(count)
                }
            )

    return {
        "text": text,
        "score": round(score, 2),
        "grade": grade["name"],
        "grade_comment": grade["comment"],
        "detected_features": detected_features,
        "vector": vector.astype(int).tolist()
    }


# ==========================================================
# 12. 최종 대화 분석
# ==========================================================

def analyze_conversation(
    selected_messages
):
    """
    선택된 사람의 전체 메시지를 분석합니다.

    핵심:

    문장별 영포티 지수
          ↓
    모든 문장의 mean()
          ↓
    최종 영포티 지수

    예:

    20
    40
    60
    80

    평균 = 50

    → 최종 영포티 지수 = 50
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
    # 문장별 영포티 지수의 평균
    # ------------------------------------------------------

    sentence_scores = [
        result["score"]
        for result in sentence_results
    ]

    final_score = float(
        np.mean(sentence_scores)
    )

    final_grade = get_grade(
        final_score
    )

    return {
        "final_score": round(
            final_score,
            2
        ),
        "final_grade": final_grade["name"],
        "final_comment": final_grade["comment"],
        "message_count": len(
            selected_messages
        ),
        "sentence_results": sentence_results
    }


# ==========================================================
# 13. 서버 상태 확인
# ==========================================================

@app.get("/")
def root():

    return {
        "service": "FORTY-SCAN",
        "status": "running",
        "model": "LogisticRegression"
    }


# ==========================================================
# 14. 화자 목록 확인 API
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

    # ------------------------------------------------------
    # TXT 확장자 확인
    # ------------------------------------------------------

    if not file.filename.lower().endswith(
        ".txt"
    ):

        raise HTTPException(
            status_code=400,
            detail="카카오톡 TXT 파일만 업로드할 수 있습니다."
        )

    # ------------------------------------------------------
    # 파일 읽기
    # ------------------------------------------------------

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    text = decode_txt(
        file_bytes
    )

    # ------------------------------------------------------
    # 카카오톡 파싱
    # ------------------------------------------------------

    messages = parse_kakao_txt(
        text
    )

    if not messages:

        raise HTTPException(
            status_code=400,
            detail="카카오톡 메시지를 찾을 수 없습니다."
        )

    # ------------------------------------------------------
    # 화자 추출
    # ------------------------------------------------------

    speakers = get_speakers(
        messages
    )

    return {
        "filename": file.filename,
        "speakers": speakers,
        "message_count": len(messages)
    }


# ==========================================================
# 15. 실제 분석 API
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
    # 1. 파일 확장자 확인
    # ------------------------------------------------------

    if not file.filename.lower().endswith(
        ".txt"
    ):

        raise HTTPException(
            status_code=400,
            detail="카카오톡 TXT 파일만 업로드할 수 있습니다."
        )

    # ------------------------------------------------------
    # 2. 파일 읽기
    # ------------------------------------------------------

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    text = decode_txt(
        file_bytes
    )

    # ------------------------------------------------------
    # 3. 카카오톡 메시지 파싱
    # ------------------------------------------------------

    messages = parse_kakao_txt(
        text
    )

    if not messages:

        raise HTTPException(
            status_code=400,
            detail="카카오톡 메시지를 찾을 수 없습니다."
        )

    # ------------------------------------------------------
    # 4. 화자 목록 확인
    # ------------------------------------------------------

    speakers = get_speakers(
        messages
    )

    # ------------------------------------------------------
    # 5. 존재하지 않는 화자 방지
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
    # 6. 선택된 화자의 메시지만 추출
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
    # 7. 분석
    # ------------------------------------------------------

    result = analyze_conversation(
        selected_messages
    )

    # ------------------------------------------------------
    # 8. 최종 결과 반환
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

        # 참고용
        "available_speakers": speakers,

        # 문장별 상세 결과
        "sentence_results": result["sentence_results"]
    }


# ==========================================================
# 실행 방법
#
# 터미널:
#
# uvicorn main:app --reload
#
# 이후:
#
# http://127.0.0.1:8000/docs
#
# ==========================================================
