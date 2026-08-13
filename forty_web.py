
# ==========================================================
# forty-scan
# 영포티 말투 판독기 - 테스트용 웹 레이어
#
# forty_backend.py 를 import 해서
# 같은 FastAPI app 에 라우트를 "추가"만 합니다.
#
# 기존 파일은 한 줄도 수정하지 않습니다.
#
# forty_backend.py 의 서버 실행은 if __name__ == "__main__" 안에 있으므로
# import 해도 서버가 뜨지 않고, app 객체와 함수들만 그대로 가져옵니다.
#
# 추가되는 것
# 1. /ui/          - 드래그앤드롭 업로드 화면 (static/index.html)
# 2. /analyze-all  - 참여자 전원을 한 번에 분석
#
# 그대로 남는 것
#   /              - 기존 상태 확인 JSON
#   /speakers      - 기존 화자 목록
#   /analyze       - 기존 화자 1명 분석
#
# 실행:
#   python forty_web.py
#   또는
#   uvicorn forty_web:app --reload
#
# 접속:
#   http://127.0.0.1:8000/ui/
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import re

from fastapi import File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

# 기존 백엔드를 그대로 재사용합니다.
#
# app                   : 이미 만들어진 FastAPI 인스턴스 (CORS 설정 포함)
# COEFFICIENTS          : 로지스틱 회귀 계수 26개 (근거 특징 정렬에 씀)
# decode_txt            : utf-8 / utf-8-sig / cp949 자동 판별
# get_speakers          : 등장 순서대로 화자 목록
# analyze_conversation  : 화자 1명의 메시지 목록 → 최종 영포티 지수
#
# parse_kakao_txt 는 여러 줄 메시지를 버리기 때문에
# 아래 섹션 2에서 따로 파서를 둡니다. (기존 함수는 그대로 살아 있습니다.)
from forty_backend import (
    COEFFICIENTS,
    analyze_conversation,
    app,
    decode_txt,
    get_speakers,
)

from forty_scan_features import FEATURE_NAMES


# ==========================================================
# 1. 기본 설정
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")

# 등급별 캐릭터 이미지 폴더입니다.
#
# charimg_web/1단계.png ~ charimg_web/5단계.png 가
# forty_scan_features.py 의 GRADES 5구간과 순서대로 짝을 이룹니다.
#
#   1단계 = 완전 청정          (0  ~ 20)
#   2단계 = 정상 범주          (20 ~ 40)
#   3단계 = 영포티 경계        (40 ~ 60)
#   4단계 = 영포티 고위험군    (60 ~ 80)
#   5단계 = 말기 영포티        (80 ~ )
#
# 원본은 charImg/ 에 있지만 장당 0.7~1.6MB(4K 해상도)라
# 결과 화면(가로 380px 안팎)에 쓰기엔 과합니다.
#
# charimg_web/ 은 640px 폭으로 줄이고 색을 256색으로 낮춘 서빙용 사본입니다.
# (레티나 2배 화면까지 커버하는 최소 크기이고, 5장 합계 6MB → 300KB 로 줄어듭니다.)
# 원본이 바뀌면 이 사본도 다시 만들어야 합니다.
#
# static 폴더 밖에 있어서 /ui 마운트로는 접근할 수 없으므로
# 아래 7번 섹션에서 /charimg 로 따로 붙입니다.

CHARIMG_DIR = os.path.join(BASE_DIR, "charimg_web")

# 로지스틱 회귀 계수(26개, FEATURE_NAMES 와 같은 순서)는
# forty_backend 에서 가져옵니다.
#
# 근거 표시에 씁니다.
# 등장 횟수만으로 정렬하면 느낌표 / 존칭_님씨 처럼
# 흔하지만 계수가 작은 특징이 상위를 차지해서 근거가 되지 못합니다.
#
# 그래서 forty_scan_test.py 의 predict_forty 와 동일하게
# 기여도 = 등장 횟수 x 계수 로 정렬합니다.

# 화자 카드 하나에 보여줄 근거 특징 개수
TOP_FEATURE_COUNT = 6

# 화자 카드 하나에 보여줄 대표 문장 개수
TOP_SENTENCE_COUNT = 3


# ==========================================================
# 2. 카카오톡 TXT 파싱
#
# 기존 forty_backend.parse_kakao_txt 에는 두 가지 문제가 있습니다.
#
# 문제 1. 형식이 하나뿐입니다.
#
#   인식:     [한병준] [오후 3:20] 아          (iOS / PC 내보내기)
#   미인식:   2024년 1월 1일 오후 3:20, 한병준 : 아   (안드로이드)
#
#   안드로이드 파일은 한 줄도 매치되지 않아
#   "카카오톡 메시지를 찾을 수 없습니다" 400 이 납니다.
#
# 문제 2. 여러 줄 메시지의 둘째 줄부터를 통째로 버립니다.
#
#   [한병준] [오후 3:20] 라떼는 말이야
#   우리 때는 이런 거 상상도 못했지~~          ← 버려짐
#   진짜 열정 하나로 버텼거든요 ^^             ← 버려짐
#
#   둘째 줄부터는 "[이름] [시각]" 접두사가 없어서 패턴에 맞지 않고,
#   기존 파서는 맞지 않는 줄을 그냥 continue 로 넘깁니다.
#
#   하필 영포티 신호(물결, ^^, 라떼_나때)가 진한 줄이 사라지는 경우가 많아
#   점수가 실제보다 낮게 나옵니다.
#
# 기존 파서를 고치지 않고, 여기서 두 문제를 모두 처리한 파서를 따로 둡니다.
# ==========================================================

# iOS / PC 내보내기
#
# 정규식은 forty_backend.parse_kakao_txt 의 것을 그대로 옮겼습니다.
# 어느 줄이 "새 메시지의 시작"인지 판정하는 기준이 갈라지면 안 됩니다.
PC_PATTERN = re.compile(
    r"^\[(?P<speaker>.*?)\]\s+"
    r"\[(?P<time>.*?)\]\s+"
    r"(?P<message>.*)$"
)

# 안드로이드 내보내기
#
#   2024년 1월 1일 오후 3:20, 한병준 : 아
#   2024. 1. 1. 오후 3:20, 한병준 : 아
ANDROID_PATTERN = re.compile(
    r"^\d{4}\s*[.년]\s*\d{1,2}\s*[.월]\s*\d{1,2}\s*[.일]?\s*"
    r"(?:오전|오후)\s*\d{1,2}:\d{2}\s*,\s*"
    r"(?P<speaker>.+?)\s*:\s+"
    r"(?P<message>.*)$"
)

# 메시지가 아닌 줄
#
# 여러 줄 메시지를 이어붙이려면 "메시지가 아닌 줄"을 먼저 걸러내야 합니다.
# 안 그러면 날짜 구분선이 바로 위 메시지에 붙어버립니다.
#
#   2026년 8월 11일 화요일          ← 날짜 구분선 (PC)
#   --------------- 2026년 8월 11일 월요일 ---------------   ← (안드로이드)
#   저장한 날짜 : 2026-08-11 15:00:00
#   한병준 님과 카카오톡 대화
SKIP_PATTERNS = [
    re.compile(r"^\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*[월화수목금토일]요일$"),
    re.compile(r"^-{3,}.*-{3,}$"),
    re.compile(r"^저장한 날짜\s*[:：]"),
    re.compile(r"님과 카카오톡 대화"),
]


def is_skippable(line):
    """메시지가 아닌 머리말 / 날짜 구분선인지 판정합니다."""

    for pattern in SKIP_PATTERNS:

        if pattern.search(line):
            return True

    return False


def parse_kakao_lines(text, pattern):
    """
    카카오톡 TXT 를 메시지 목록으로 만듭니다.

    pattern 에 맞는 줄     → 새 메시지 시작
    맞지 않는 줄           → 바로 앞 메시지의 둘째 줄 이후로 이어붙임
    머리말 / 날짜 구분선   → 버림

    카카오톡에서 여러 줄 메시지는 실제로 "한 번에 보낸 메시지 하나"이므로
    (Shift+Enter 또는 붙여넣기) 줄을 나누지 않고 하나로 합칩니다.
    문장 1개 = 샘플 1개 라는 기존 규칙과도 맞습니다.

    반환 형식은 parse_kakao_txt 와 동일합니다.

    [
        {
            "speaker": "한병준",
            "message": "라떼는 말이야\n우리 때는 이런 거 상상도 못했지~~"
        }
    ]
    """

    messages = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        match = pattern.match(line)

        # --------------------------------------------------
        # 새 메시지
        # --------------------------------------------------

        if match:

            speaker = match.group(
                "speaker"
            ).strip()

            message = match.group(
                "message"
            ).strip()

            if not speaker:
                continue

            messages.append(
                {
                    "speaker": speaker,
                    "message": message
                }
            )

            continue

        # --------------------------------------------------
        # 메시지가 아닌 줄
        # --------------------------------------------------

        if is_skippable(line):
            continue

        # --------------------------------------------------
        # 여러 줄 메시지의 둘째 줄 이후
        #
        # 앞선 메시지가 없으면 (파일 첫머리) 그냥 버립니다.
        # --------------------------------------------------

        if not messages:
            continue

        previous = messages[-1]["message"]

        if previous:
            messages[-1]["message"] = previous + "\n" + line
        else:
            messages[-1]["message"] = line

    # 내용이 빈 메시지는 제외합니다. (기존 파서와 동일한 규칙)
    return [
        item
        for item in messages
        if item["message"]
    ]


def parse_kakao_any(text):
    """
    두 가지 카카오톡 내보내기 형식을 모두 처리합니다.

    1. iOS / PC 형식을 먼저 시도합니다.
    2. 결과가 0건일 때만 안드로이드 형식을 시도합니다.
    """

    messages = parse_kakao_lines(
        text,
        PC_PATTERN
    )

    if messages:
        return messages

    return parse_kakao_lines(
        text,
        ANDROID_PATTERN
    )


# ==========================================================
# 3. 업로드 파일 → 메시지 목록
#
# forty_backend.load_messages 와 구조는 같지만
# 파서만 parse_kakao_any 로 바꾼 버전입니다.
#
# 기존 load_messages 는 내부에서 parse_kakao_txt 를 직접 부르기 때문에
# 파서만 갈아끼울 수가 없어서 여기에 따로 둡니다.
# ==========================================================

async def load_messages_web(file):
    """
    업로드 파일을 검증하고 카카오톡 메시지 목록으로 만듭니다.

    확장자 확인 → 읽기 → 인코딩 판별 → 파싱 → 빈 결과 확인
    """

    # filename 은 None 일 수 있으므로 그대로 .lower() 를 부르지 않습니다.
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

    messages = parse_kakao_any(
        text
    )

    if not messages:

        raise HTTPException(
            status_code=400,
            detail=(
                "카카오톡 메시지를 찾을 수 없습니다. "
                "카카오톡 '대화 내보내기'로 저장한 TXT 파일인지 확인해 주세요."
            )
        )

    return messages


# ==========================================================
# 4. 근거 집계
#
# analyze_sentence 가 문장마다
#
#   detected_features = [{"name", "count", "is_fuzzy"}, ...]
#
# 를 이미 돌려주므로, 화자 단위로 합산하기만 하면 됩니다.
# ==========================================================

def aggregate_features(sentence_results):
    """
    화자 1명의 문장별 결과에서 근거가 되는 특징을 뽑습니다.

    기여도 = 총 등장 횟수 x 로지스틱 회귀 계수

    계수가 양수면 영포티 지수를 올린 특징 (영포티 신호 24개)
    계수가 음수면 영포티 지수를 내린 특징 (젊은 세대 신호 5개)

    기여도 절댓값 내림차순으로 상위 TOP_FEATURE_COUNT 개를 반환합니다.
    """

    total_counts = {}

    fuzzy_names = set()

    for result in sentence_results:

        for feature in result["detected_features"]:

            name = feature["name"]

            total_counts[name] = (
                total_counts.get(name, 0)
                + feature["count"]
            )

            if feature["is_fuzzy"]:

                fuzzy_names.add(name)

    coefficient_of = dict(
        zip(FEATURE_NAMES, COEFFICIENTS)
    )

    features = []

    for name, count in total_counts.items():

        coefficient = float(
            coefficient_of.get(name, 0.0)
        )

        features.append(
            {
                "name": name,
                "count": int(count),
                "coef": round(coefficient, 4),
                "contribution": round(
                    count * coefficient,
                    4
                ),
                "is_fuzzy": name in fuzzy_names
            }
        )

    features.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True
    )

    return features[:TOP_FEATURE_COUNT]


# ==========================================================
# 5. 대표 문장 추출
# ==========================================================

def pick_top_sentences(sentence_results):
    """
    영포티 지수가 가장 높은 문장 몇 개를 뽑습니다.

    문장별 결과를 전부 내보내면 대화가 길 때 응답이 수 MB가 되므로
    상위 TOP_SENTENCE_COUNT 개만 담습니다.

    vector(26차원)는 화면에 쓰지 않으므로 제외합니다.
    """

    ordered = sorted(
        sentence_results,
        key=lambda item: item["score"],
        reverse=True
    )

    top = []

    for result in ordered[:TOP_SENTENCE_COUNT]:

        top.append(
            {
                "text": result["text"],
                "score": result["score"],
                "grade": result["grade"],
                "detected_features": result["detected_features"]
            }
        )

    return top


# ==========================================================
# 6. 참여자 전원 분석 API
# ==========================================================

@app.post("/analyze-all")
async def analyze_all(
    file: UploadFile = File(...)
):
    """
    카카오톡 TXT 를 받아 대화에 등장한 모든 화자를 한 번에 분석합니다.

    화자마다 기존 analyze_conversation 을 그대로 호출하므로
    /analyze 로 화자 1명씩 돌린 결과와 점수가 정확히 같습니다.

    반환:

        filename
        message_count
        results  (final_score 내림차순)
    """

    # ------------------------------------------------------
    # 1. 업로드 파일 → 카카오톡 메시지 목록
    # ------------------------------------------------------

    messages = await load_messages_web(
        file
    )

    # ------------------------------------------------------
    # 2. 화자 목록
    # ------------------------------------------------------

    speakers = get_speakers(
        messages
    )

    # ------------------------------------------------------
    # 3. 화자별 분석
    # ------------------------------------------------------

    results = []

    for speaker in speakers:

        selected_messages = [
            item["message"]
            for item in messages
            if item["speaker"] == speaker
        ]

        if not selected_messages:
            continue

        # 점수 계산은 기존 함수를 그대로 씁니다.
        analyzed = analyze_conversation(
            selected_messages
        )

        sentence_results = analyzed["sentence_results"]

        results.append(
            {
                "speaker": speaker,

                "final_score": analyzed["final_score"],
                "final_grade": analyzed["final_grade"],
                "final_comment": analyzed["final_comment"],

                "message_count": analyzed["message_count"],

                # 표본이 얼마나 되는지
                #
                # reliability     : 0(근거 없음) ~ 1(충분)
                # is_provisional  : True 면 등급이 "표본 부족" 이고 판정 보류
                "reliability": analyzed["reliability"],
                "is_provisional": analyzed["is_provisional"],

                # 근거
                "top_features": aggregate_features(
                    sentence_results
                ),

                # 대표 문장
                "top_sentences": pick_top_sentences(
                    sentence_results
                )
            }
        )

    if not results:

        raise HTTPException(
            status_code=400,
            detail="분석할 화자를 찾을 수 없습니다."
        )

    # ------------------------------------------------------
    # 4. 영포티 지수 높은 순으로 정렬
    # ------------------------------------------------------

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return {
        "filename": file.filename,
        "message_count": len(messages),
        "results": results
    }


# ==========================================================
# 7. 정적 파일 서빙
#
# 기존 @app.get("/") 이 이미 상태 확인 JSON 을 반환하고
# 먼저 등록되어 있으므로 "/" 는 덮어쓸 수 없습니다.
# (FastAPI 는 등록된 순서대로 경로를 매칭합니다.)
#
# 그래서 충돌이 없는 /ui 에 화면을 붙입니다.
#
# html=True 로 두면 /ui/ 요청에 index.html 을 돌려줍니다.
# ==========================================================

if not os.path.isdir(STATIC_DIR):

    raise FileNotFoundError(
        f"\n화면 파일 폴더를 찾을 수 없습니다.\n"
        f"경로:\n{STATIC_DIR}\n"
    )


app.mount(
    "/ui",
    StaticFiles(
        directory=STATIC_DIR,
        html=True
    ),
    name="ui"
)


# 등급별 캐릭터 이미지입니다.
#
# 화면에서는 점수 구간을 1~5 로 바꾼 뒤
#
#   /charimg/1단계.png ~ /charimg/5단계.png
#
# 으로 부릅니다. (한글 파일명은 브라우저가 알아서 인코딩합니다.)

if not os.path.isdir(CHARIMG_DIR):

    raise FileNotFoundError(
        f"\n캐릭터 이미지 폴더를 찾을 수 없습니다.\n"
        f"경로:\n{CHARIMG_DIR}\n"
    )


app.mount(
    "/charimg",
    StaticFiles(
        directory=CHARIMG_DIR
    ),
    name="charimg"
)


# ==========================================================
# 실행 방법
#
# 터미널:
#
# python forty_web.py
#
# 또는:
#
# uvicorn forty_web:app --reload
#
# 이후:
#
# http://127.0.0.1:8000/ui/     ← 판독 화면
# http://127.0.0.1:8000/docs    ← API 문서
#
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "forty_web:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
