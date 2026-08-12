# syntax=docker/dockerfile:1

# ==========================================================
# forty-scan - Google Cloud Run 배포용 이미지
#
# 기존 파이썬 파일은 한 줄도 수정하지 않습니다.
#
# forty_web.py 의 uvicorn.run(host="127.0.0.1", port=8000) 은
# if __name__ == "__main__" 가드 안에 있어서,
# 아래 CMD 처럼 uvicorn 이 forty_web 을 "import" 하는 방식으로 띄우면
# 실행되지 않습니다. 그래서 코드 수정 없이 Cloud Run 규격
# (0.0.0.0:$PORT) 을 맞출 수 있습니다.
# ==========================================================

# Cloud Run 은 linux/amd64 만 실행합니다.
FROM --platform=linux/amd64 python:3.12-slim

# LANG 은 베이스 이미지가 이미 C.UTF-8 로 설정하지만,
# 베이스가 바뀌어도 한글 파일명이 깨지지 않도록 명시합니다.
#   - model/forty_scan_현경_model.pkl
#   - charimg_web/1단계.png ~ 5단계.png
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

# ----------------------------------------------------------
# 1단계: 의존성만 먼저 설치 (레이어 캐시)
#
# requirements.txt 가 안 바뀌면 소스만 고쳐도 이 레이어는 재사용됩니다.
#
# --only-binary=:all: 를 주는 이유
#   numpy / scipy / scikit-learn 은 cp312 manylinux 휠이 전부 존재합니다.
#   혹시 휠이 없으면 pip 이 조용히 소스 빌드(40분+, 컴파일러 필요)를
#   시도하다 실패하는데, 이 옵션을 주면 5초 만에 명확하게 실패합니다.
# ----------------------------------------------------------
COPY requirements.txt ./
RUN pip install --only-binary=:all: --no-cache-dir -r requirements.txt

# ----------------------------------------------------------
# 2단계: 소스 복사
# .dockerignore 가 학습 데이터 / 원본 아트 / 문서를 걸러냅니다.
# ----------------------------------------------------------
COPY . ./

# ----------------------------------------------------------
# 3단계: 비루트 사용자로 실행
# ----------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 문서용 표시일 뿐입니다. 실제 포트는 Cloud Run 이 $PORT 로 주입합니다.
EXPOSE 8080

# ==========================================================
# 반드시 이 형태여야 합니다.
#
#   sh -c   → ${PORT} 가 확장됩니다.
#             CMD ["uvicorn", ..., "--port", "$PORT"] 처럼 쓰면
#             exec 형식이라 셸 확장이 일어나지 않고,
#             uvicorn 이 리터럴 문자열 "$PORT" 를 받아 죽습니다.
#             (Cloud Run 초보 실패 원인 1위)
#
#   exec    → uvicorn 이 PID 1 이 되어 SIGTERM 을 직접 받습니다.
#             (리비전 교체 시 graceful shutdown)
#
#   :-8080  → 로컬에서 -e PORT 없이 docker run 해도 동작합니다.
#
#   --workers 1
#           → Cloud Run 은 인스턴스 단위로 스케일합니다.
#             컨테이너 안 워커를 늘리면 워커당 RAM 만 200MB 씩 더 씁니다.
#
#   --timeout-keep-alive 65
#           → Cloud Run 프론트엔드가 재사용하려던 커넥션을
#             서버가 먼저 닫아 생기는 간헐적 502 를 막습니다.
# ==========================================================
CMD ["sh", "-c", "exec uvicorn forty_web:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 65"]
