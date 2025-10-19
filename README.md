# TwoOfUsTunes
두 사람의 감정(Valence–Arousal)을 결합해 **공통 분위기의 플레이리스트**를 추천하는 ML 엔진

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research--Prototype-orange)

## 1) What & Why
- **문제**: 그날의 기분(감정)이 다르면 같이 들을 음악 고르기 어렵다.
- **해결**: 사용자의 감정을 Russell **Valence–Arousal(±1)** 좌표로 받아, 곡의 감정 벡터와 매칭해 Top-K를 추천.
- **2인 모드**: 두 감정 벡터를 평균/교집합/합집합 전략으로 **공통 무드** 플레이리스트 생성.

## 2) Core Features
- 🧭 **감정 입력 융합**: 슬라이더 + (옵션) 컨텍스트/행동 로그 + SER/FER 플러그인
- 🎯 **콘텐츠 기반 추천**: `(valence, arousal, tempo, energy, danceability)` 임베딩 유사도
- 🤝 **두 사람 추천**: `avg | intersect | union` 전략 + 다양성 제약
- 🔁 **액티브 러닝 훅**: 스킵/좋아요/재생시간 → 모델 업데이트 포인트 제공
- 📊 **오프라인 지표**: NDCG@K, Hit@K, 다양성(장르 엔트로피)

## 3) Data (예시)
- **DEAM / MoodyLyrics** (기본) + **PMEmo**(선택)
- 표준 스키마:  
  `track_id, title, artist, va:(v,a), audio_feats:Vector, lyric_emb:Vector, meta:{genre, tempo, energy, danceability, ...}`

## 4) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 5) Quickstart (CLI)

```bash
# 피처 추출(예시)
python features/extract_audio.py  --in data/raw --out data/processed/tracks.parquet
python features/extract_lyric.py  --in data/raw --out data/processed/lyrics.parquet

# 1인 추천
python cli.py recommend --v 0.6 --a 0.2 --topk 20 --out results/v1_rec.csv

# 2인 추천
python cli.py recommend-two \
  --v1 0.7 --a1 0.4 --v2 -0.2 --a2 0.6 \
  --strategy avg --w 0.5 --topk 20 --out results/v2_rec.csv

# 오프라인 평가
python eval/offline_metrics.py --rec results/v1_rec.csv --gt data/processed/ground_truth.csv
```

## 6) Engine API (파이썬)

```python
from engine.recommender import recommend, rerank_two_users
from engine.state_estimator import estimate

# 감정 추정
E = estimate({"slider": {"v": 0.5, "a": -0.1}, "context": {"hour": 21, "weather": "rain"}})

# 1인 추천
items = recommend(state={"v": 0.5, "a": -0.1}, topk=20)

# 2인 추천
items2 = rerank_two_users(e_a=(0.7,0.4), e_b=(-0.2,0.6), strategy="avg", w=0.5, topk=20)
```

## 7) Metrics

* **추천**: NDCG@10, Hit@10
* **정확도**(옵션 회귀/분류): MSE/RMSE, Acc/Precision/Recall
* **다양성**: 장르 엔트로피/아티스트 커버리지

## 8) Repo Structure

아래 “역할별 파일 구조” 참고.

## 9) License

MIT

---

# 🗂 역할별 파일 구조 (책임자 표기)

```text
TwoOfUsTunes/
├─ data/                                   # [데이터·피처]
│  ├─ raw/                                 #  원본 (gitignore/LFS)
│  └─ processed/                           #  표준화 결과: tracks.parquet 등
│
├─ features/                               # [데이터·피처]
│  ├─ extract_audio.py                     #  librosa 등으로 MFCC/Chroma/Tempo 추출
│  ├─ extract_lyric.py                     #  TF-IDF/BERT 임베딩
│  └─ build_index.py                       #  (옵션) FAISS/Annoy 인덱스 생성
│
├─ engine/                                 # [모델·추천]
│  ├─ recommender.py                       #  콘텐츠 기반 Top-K, 다양성, 2인 재랭킹
│  ├─ blend.py                             #  avg/intersect/union 전략, 가중치 w
│  ├─ state_estimator.py                   #  [센싱/퓨전] 입력→(v,a,uncertainty)
│  ├─ feedback.py                          #  [센싱/퓨전] 로그→보상/업데이트 훅
│  ├─ models/                              #  (옵션) 회귀/분류 저장/로딩
│  │  ├─ va_regressor.pkl
│  │  └─ quadrant_clf.pkl
│  └─ utils.py
│
├─ sensing/                                # [센싱/퓨전] (옵션 플러그인)
│  ├─ ser.py                               #  음성 감정(MFCC+SVM 등), 로컬 처리
│  └─ fer.py                               #  표정 감정(경량 CNN), 동의 필수
│
├─ eval/                                   # [모델·추천] + [실험·평가]
│  ├─ offline_metrics.py                   #  NDCG/Hit/다양성 계산
│  ├─ plots.py                             #  VA 맵/분포/지표 그래프
│  └─ ab_simulator.py                      #  (옵션) 전략 비교 시뮬
│
├─ notebooks/                              # [실험·평가]
│  ├─ 01_eda.ipynb                         #  데이터 분포/품질
│  ├─ 02_train_va_regressor.ipynb          #  (옵션) VA 회귀/분류 학습
│  └─ 03_eval_recommender.ipynb            #  추천 성능/다양성 평가
│
├─ experiments/                            # [실험·평가·PM]
│  ├─ configs/                             #  실험 설정(yaml/json)
│  ├─ logs/                                #  결과 로그/지표
│  └─ results/                             #  CSV 출력 및 스냅샷
│
├─ results/                                # [실험·평가·PM] (사용자 실행 산출)
│  ├─ v1_rec.csv
│  └─ v2_rec.csv
│
├─ tests/                                  # [전체 품질]
│  ├─ test_recommender.py
│  ├─ test_state_estimator.py
│  └─ test_utils.py
│
├─ cli.py                                  # [실험·평가·PM] 로컬 실행 CLI
├─ requirements.txt                        #  핵심 의존성
├─ README.md                               #  위 템플릿으로 생성
└─ .gitignore
```

---

## 🔧 각 역할별 “책임 파일” 요약

### 1) 데이터·피처링 엔지니어

* `data/raw/*` → `data/processed/tracks.parquet`
* `features/extract_audio.py`, `features/extract_lyric.py`, `features/build_index.py`
* 품질 리포트: `notebooks/01_eda.ipynb`, `reports/data_qa.md`(원하면 폴더 추가)

### 2) 모델·추천 엔지니어

* `engine/recommender.py`, `engine/blend.py`, `eval/offline_metrics.py`, `eval/plots.py`
* (옵션) `engine/models/*` 저장/불러오기

### 3) 입력·감정추정(Sensing/Fusion)

* `engine/state_estimator.py`, `engine/feedback.py`
* (옵션) `sensing/ser.py`, `sensing/fer.py` (로컬/on-device 전제)

### 4) 실험·평가·UX/PM

* `cli.py` (아래 스켈레톤 예시)
* `notebooks/*`, `experiments/*`, `results/*`, `README.md` 편집/정리

---

## 🧪 `cli.py` 스켈레톤 (간단 예시)

```python
import argparse
from engine.recommender import recommend, rerank_two_users
from engine.state_estimator import estimate


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    r1 = sub.add_parser("recommend")
    r1.add_argument("--v", type=float, required=True)
    r1.add_argument("--a", type=float, required=True)
    r1.add_argument("--topk", type=int, default=20)
    r1.add_argument("--out", type=str, default="results/v1_rec.csv")

    r2 = sub.add_parser("recommend-two")
    r2.add_argument("--v1", type=float, required=True)
    r2.add_argument("--a1", type=float, required=True)
    r2.add_argument("--v2", type=float, required=True)
    r2.add_argument("--a2", type=float, required=True)
    r2.add_argument("--strategy", choices=["avg","intersect","union"], default="avg")
    r2.add_argument("--w", type=float, default=0.5)
    r2.add_argument("--topk", type=int, default=20)
    r2.add_argument("--out", type=str, default="results/v2_rec.csv")

    args = parser.parse_args()

    if args.cmd == "recommend":
        items = recommend(state={"v": args.v, "a": args.a}, topk=args.topk)
        # TODO: save to args.out
    elif args.cmd == "recommend-two":
        items = rerank_two_users(
            e_a=(args.v1, args.a1),
            e_b=(args.v2, args.a2),
            strategy=args.strategy, w=args.w, topk=args.topk
        )
        # TODO: save to args.out
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

---

## 📄 `.gitignore` 예시

```
# data & artifacts
data/raw/*
model_artifacts/*
results/*.csv

# notebooks
*.ipynb_checkpoints

# audio
*.mp3
*.wav

# env
.venv/
__pycache__/
*.pyc
```

---

## 📦 `requirements.txt` (예시)

```
numpy
pandas
scikit-learn
librosa
faiss-cpu     # 인덱스 사용 시
matplotlib
scipy
tqdm
```
