
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

    final_grade, final_comment = get_grade(
        final_score
    )

    return {
        "final_score": round(
            final_score,
            2
        ),
        "final_grade": final_grade,
        "final_comment": final_comment,
        "message_count": len(
            selected_messages
        ),
        "sentence_results": sentence_results
    }


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
