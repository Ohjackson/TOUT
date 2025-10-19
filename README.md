아래 내용을 그대로 `README.md`로 붙여 쓰면 돼. 필요하면 나중에 배지/이미지 추가해줄게.

---

# TwoOfUsTunes

두 사람의 감정(Valence–Arousal)을 결합해 **공통 분위기의 플레이리스트**를 추천하는 머신러닝 엔진

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Research--Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1. 서비스 목적 & 흐름

### 목적

* 그날그날 **감정 상태(Valence–Arousal, ±1 범위)**에 맞는 음악을 추천
* **2인 모드**에서 두 사람의 감정을 조합해 “둘이 들을” 공통 분위기 플레이리스트 생성
* 행동 피드백(스킵/좋아요/완청)을 받아 **점점 개인화**되는 추천

### 기본 흐름

1. 사용자 입력 → 슬라이더(Valence, Arousal) + (선택) 컨텍스트/행동 로그
2. `state_estimator`가 감정 벡터 **E=(v,a)** 및 불확실성 산출
3. `recommender`가 곡 임베딩(va, tempo, energy, danceability 등)과 **유사도 기반 Top-K** 추천
4. (2인) `blend` 전략(**avg / intersect / union**)으로 공통 리스트 재랭킹
5. 사용 피드백을 `feedback`에서 수집 → 액티브 러닝 훅으로 업데이트 포인트 제공

---

## 2. 기술 스택

* **언어**: Python 3.10+
* **ML/데이터**: NumPy, Pandas, scikit-learn, (선택) FAISS, librosa
* **시각화/평가**: Matplotlib
* **실험**: Jupyter Notebook
* **CLI/도구**: argparse, tqdm
* **패키징**: `requirements.txt` (또는 추후 `pyproject.toml` 전환 가능)

---

## 3. 설치 & 로컬 개발

### 요구 사항

* Python **3.10+**
* (오디오 피처 사용 시) **ffmpeg**, **librosa**가 사용하는 시스템 라이브러리
* macOS/Windows/Linux 모두 지원

### 설치

```bash
git clone https://github.com/<your-org>/TwoOfUsTunes.git
cd TwoOfUsTunes

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 데이터 준비(예시)

```bash
# 오디오/가사 피처 추출 (원본은 data/raw/* 에 배치)
python features/extract_audio.py  --in data/raw --out data/processed/tracks.parquet
python features/extract_lyric.py  --in data/raw --out data/processed/lyrics.parquet
```

### 실행 예시 (CLI)

```bash
# 1인 추천
python cli.py recommend --v 0.6 --a 0.2 --topk 20 --out results/v1_rec.csv

# 2인 추천
python cli.py recommend-two \
  --v1 0.7 --a1 0.4 --v2 -0.2 --a2 0.6 \
  --strategy avg --w 0.5 --topk 20 --out results/v2_rec.csv

# 오프라인 평가 (예: 추천 결과 vs. 내부 GT)
python eval/offline_metrics.py --rec results/v1_rec.csv --gt data/processed/ground_truth.csv
```

---

## 4. 파일 구조

```text
TwoOfUsTunes/
├─ data/
│  ├─ raw/                      # 원본 데이터 (gitignore 또는 LFS)
│  └─ processed/                # 표준화된 데이터셋 (tracks.parquet 등)
├─ features/
│  ├─ extract_audio.py          # 오디오 피처: MFCC/Chroma/Tempo...
│  ├─ extract_lyric.py          # 가사 임베딩: TF-IDF/BERT 등
│  └─ build_index.py            # (선택) FAISS/Annoy 인덱스
├─ engine/
│  ├─ recommender.py            # Top-K 추천, 다양성 제약, 재랭킹
│  ├─ blend.py                  # 2인 전략: avg/intersect/union
│  ├─ state_estimator.py        # 입력 융합 → (v,a,uncertainty)
│  ├─ feedback.py               # 스킵/좋아요/완청 로그 처리
│  ├─ models/                   # (선택) 회귀/분류 모델 아티팩트
│  └─ utils.py
├─ sensing/                     # (선택) 입력 플러그인
│  ├─ ser.py                    # 음성 감정(SER) - 로컬 처리
│  └─ fer.py                    # 표정 감정(FER) - 동의 필수
├─ eval/
│  ├─ offline_metrics.py        # NDCG/Hit/다양성 계산
│  ├─ plots.py                  # VA 맵/분포/지표 그래프
│  └─ ab_simulator.py           # (선택) 전략 비교 시뮬레이터
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_train_va_regressor.ipynb
│  └─ 03_eval_recommender.ipynb
├─ experiments/                 # 실험 설정/로그/결과
│  ├─ configs/
│  ├─ logs/
│  └─ results/
├─ results/                     # 사용자 실행 산출(csv 등)
├─ tests/
│  ├─ test_recommender.py
│  ├─ test_state_estimator.py
│  └─ test_utils.py
├─ cli.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

`.gitignore` 예시

```
data/raw/*
model_artifacts/*
results/*.csv
.venv/
__pycache__/
*.pyc
*.ipynb_checkpoints
*.mp3
*.wav
```

---

## 5. 브랜치 전략

* 기본 브랜치: **`main`** (안정 릴리스)
* 개발 브랜치: **`dev`** (기본 머지 대상)
* 기능 브랜치: `feature/<키워드>`
  예) `feature/recommender-topk`, `feature/ser-plugin`
* 버그픽스: `fix/<키워드>`
* 문서: `docs/<키워드>`
* 실험: `exp/<키워드>` (실험용 코드/노트북)

> 규칙: 기능/수정은 **반드시 PR로 `dev`** 에 머지. 검증 후 주기적으로 `main`으로 릴리스.

---

## 6. 커밋 컨벤션 (Conventional Commits)

형식:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**type 예시**

* `feat` 새 기능
* `fix` 버그 수정
* `docs` 문서 수정
* `refactor` 리팩터링(기능 변경 없음)
* `perf` 성능 개선
* `test` 테스트 관련
* `build` 빌드/의존성
* `chore` 자잘한 변경

**예시**

```
feat(recommender): add top-k cosine similarity ranking
fix(state): handle empty context gracefully
docs(readme): update local run commands
```

---

## 7. PR 규칙 & 템플릿

### 규칙

* 대상 브랜치: 기본 **`dev`**
* 리뷰어 최소 1명 승인 필요
* CI 통과(테스트/포맷) 필수
* PR 단위는 작게, 스쿼시 머지 권장

### 템플릿 ( `.github/PULL_REQUEST_TEMPLATE.md` )

```markdown
## 목적
- (이 PR이 해결하는 문제/추가 기능을 한 줄로)

## 주요 변경
- (핵심 변경 요약)
- (API/CLI 시그니처 변경 시 표기)

## 체크리스트
- [ ] 테스트 추가/수정
- [ ] 로컬에서 기본 실행 확인 (`cli.py` quickstart)
- [ ] 문서/README 반영
- [ ] 호환성 이슈 없음

## 스크린샷/로그(선택)
- (결과 그래프, CLI 출력, 성능 수치 등)
```

---

## 8. 개발 규칙(코드 스타일 & 테스트)

* **PEP8** 준수, 타입힌트 권장
* 함수/모듈 **docstring** 필수(요약/입출력/예외)
* 핵심 로직 유닛테스트 필수(`tests/`)
* 큰 데이터/모델 아티팩트는 **커밋 금지** → `data/raw/*`, `model_artifacts/*`는 gitignore

---

## 9. 예시 API (파이썬)

```python
from engine.recommender import recommend, rerank_two_users
from engine.state_estimator import estimate

E = estimate({"slider": {"v": 0.5, "a": -0.1}, "context": {"hour": 21, "weather": "rain"}})

items = recommend(state={"v": 0.5, "a": -0.1}, topk=20)

duet = rerank_two_users(
    e_a=(0.7, 0.4),
    e_b=(-0.2, 0.6),
    strategy="avg",
    w=0.5,
    topk=20
)
```

---

## 10. Requirements (의존성)

`requirements.txt` (예시)

```
numpy
pandas
scikit-learn
librosa
faiss-cpu     # (선택) 대용량 인덱싱에 사용
matplotlib
scipy
tqdm
```

---

## 11. 라이선스

```
MIT License

Copyright (c) 2025 TwoOfUsTunes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

필요하면 위 PR 템플릿/`.gitignore` 파일도 같이 만들어줄게. README에 **배지/로고**나 **VA 좌표 이미지** 추가할 거면 말만 해!
