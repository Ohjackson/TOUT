

# NVify: The Next-Vibe Recommender


> **"Your emotional spectrum, in a perfect tune."**
>
> NVify는 사용자의 현재 감정 상태뿐만 아니라, 그 감정을 다루고 싶은 내면의 '의도(Intent)'까지 파악하여 음악을 추천하는 차세대 감성 지능형 추천 시스템입니다.

## 목차 (Table of Contents)

1.  [**프로젝트 개요 (Overview)**](#1-프로젝트-개요-overview)
2.  [**핵심 기능 (Core Features)**](#2-핵심-기능-core-features)
3.  [**시스템 아키텍처 (System Architecture)**](#3-시스템-아키텍처-system-architecture)
4.  [**동작 원리 (How It Works)**](#4-동작-원리-how-it-works)
5.  [**시작하기 (Getting Started)**](#5-시작하기-getting-started)
    *   [사전 요구사항](#사전-요구사항)
    *   [설치 가이드](#설치-가이드)
6.  [**사용 방법 (Usage)**](#6-사용-방법-usage)
7.  [**프로젝트 구조 (Project Structure)**](#7-프로젝트-구조-project-structure)
8.  [**개발 로드맵 (Development Roadmap)**](#8-개발-로드맵-a-to-z)
9.  [**기여하기 (Contributing)**](#9-기여하기-contributing)
10. [**라이선스 (License)**](#10-라이선스-license)

---

### 1. 프로젝트 개요 (Overview)

기존의 음악 추천 시스템은 주로 사용자의 과거 청취 기록을 기반으로 유사한 패턴의 음악을 추천합니다. 하지만 사람의 음악적 니즈는 고정되어 있지 않으며, 현재의 감정과 상황에 따라 역동적으로 변화합니다.

**NVify**는 이러한 문제점을 해결하기 위해 탄생했습니다. 저희는 심리학적 감정 모델(Valence-Arousal)을 통해 음악의 감성적 DNA를 분석하고, 사용자가 자신의 감정을 **'더 깊이 느끼고 싶은지(몰입)'** 또는 **'벗어나고 싶은지(전환)'**와 같은 '의도'를 직접 선택하게 함으로써, 단순한 예측을 넘어 사용자와 깊이 교감하는 추천 경험을 제공합니다.

### 2. 핵심 기능 (Core Features)

*   **🧠 2-Step 감성 입력:** 현재 '감정'과 감정을 다룰 '의도'를 2단계에 걸쳐 선택하여 복잡한 내면 상태를 정확하게 전달합니다.
*   🎶 **하이브리드 추천 모델:** 사용자의 장기적 취향(협업 필터링)과 음악의 고유한 감성 속성(콘텐츠 기반 필터링)을 하나의 잠재 공간에서 동시에 학습합니다.
*   🚀 **랭킹 최적화:** 추천 목록의 상위권 정확도를 직접적으로 최적화하는 WARP Loss를 사용하여, 사용자가 만족할 만한 실용적인 추천 결과를 생성합니다.
*   📊 **데이터 기반 설계:** 두 개의 대규모 공개 데이터셋을 병합하고 정제하여 실제 사용 패턴과 음악적 특성을 반영합니다.

### 3. 시스템 아키텍처 (System Architecture)

NVify는 다음과 같은 단계적 파이프라인으로 구성되어 있습니다.

```mermaid
graph TD
    subgraph 1. 데이터 계층 (Data Layer)
        A[Spotify-Last.fm 데이터셋]
        B[Last.fm-VADS 데이터셋]
    end

    subgraph 2. 데이터 전처리 및 통합 (Preprocessing)
        C[데이터 정제 및 병합]
        D[특징 공학: V-A 범주화]
    end

    subgraph 3. 모델링 (Modeling)
        E[하이브리드 모델 학습<br>(WARP Loss)]
        F[학습된 모델 저장]
    end

    subgraph 4. 추천 서빙 (Serving)
        G{사용자 입력<br>(감정 + 의도)}
        H[개인화 랭킹]
        I[감성 의도 기반 재랭킹]
        J[최종 플레이리스트 생성]
    end

    A & B --> C --> D --> E --> F
    G --> H --> I --> J
```

### 4. 동작 원리 (How It Works)

*   **핵심 모델:** 잠재 요인(Latent Factor) 기반 하이브리드 모델
    *   사용자와 노래를 단순히 ID로만 보지 않고, 관련된 모든 특징(ID, 장르, 아티스트, 감성 카테고리 등)들의 **임베딩 벡터 합**으로 표현합니다. 이를 통해 사용자의 취향과 노래의 속성 간의 복잡한 관계를 하나의 잠재 공간에서 학습합니다.
*   **학습 알고리즘:** WARP (Weighted Approximate-Rank Pairwise) Loss
    *   "사용자가 좋아한 노래의 예측 점수는, 싫어한 노래의 예측 점수보다 높아야 한다"는 원칙을 기반으로 학습합니다. 특히, 맞추기 어려운 상위권 순위를 올바르게 예측했을 때 더 큰 가중치를 부여하므로, 추천 목록의 실용적인 품질을 극대화합니다.

### 5. 시작하기 (Getting Started)

#### 사전 요구사항
*   [Anaconda](https://www.anaconda.com/products/distribution) 또는 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
*   Python 3.9 이상

#### 설치 가이드

1.  **프로젝트 복제 (Clone):**
    ```bash
    git clone https://github.com/AhnJaehyun/NVify.git
    cd NVify
    ```

2.  **Conda 가상환경 생성 및 활성화:**
    ```bash
    conda create -n nvify-env python=3.9
    conda activate nvify-env
    ```

3.  **필수 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```
    > `requirements.txt` 파일에는 `pandas`, `numpy`, `scikit-learn`, `lightfm` 등 프로젝트에 필요한 모든 라이브러리가 명시되어 있습니다.

4.  **데이터 다운로드:**
    *   `data/` 폴더에 제안서에 명시된 두 데이터셋을 다운로드하여 위치시킵니다.

### 6. 사용 방법 (Usage)

프로젝트의 핵심 기능을 체험해볼 수 있는 데모 스크립트를 실행하는 방법입니다.

1.  **데이터 전처리 및 모델 학습 실행 (최초 1회):**
    > 이 과정은 데이터셋의 크기에 따라 수십 분 이상 소요될 수 있습니다.
    ```bash
    python main.py --train
    ```

2.  **추천 데모 실행:**
    ```bash
    python main.py --recommend
    ```    *   스크립트를 실행하면 다음과 같은 프롬프트가 나타납니다. 안내에 따라 번호를 입력하여 추천을 받아보세요.

    ```
    ========================================
    🎵 Welcome to NVify Recommender! 🎵
    ========================================
    먼저, 당신의 User ID를 입력해주세요 (예: 2): 2

    --- STEP 1: 지금 어떤 감정이신가요? ---
    1: 😊 기쁨
    2: 😢 슬픔
    3: 🔥 열정/신남
    4: 😌 평온/이완
    당신의 선택: 2

    --- STEP 2: 이 감정을 어떻게 경험하고 싶으세요? ---
    1: 🌊 몰입하기 (감정에 더 깊이 빠져들기)
    2: ☀️ 전환하기 (감정에서 벗어나 기분 전환하기)
    당신의 선택: 1

    ----------------------------------------
    '슬픔'에 '몰입'하기 위한 당신만의 플레이리스트입니다:
    ----------------------------------------
    1. Artist A - Sad Song Title 1
    2. Artist B - Melancholy Melody
    3. Artist C - Raindrop Ballad
    ...
    ```

### 7. 프로젝트 구조 (Project Structure)

```
NVify/
├── data/                  # 원본 및 전처리된 데이터
│   ├── raw/               # 다운로드한 원본 데이터
│   └── processed/         # 정제 및 통합된 마스터 데이터셋
├── notebooks/             # 데이터 탐색 및 실험용 Jupyter Notebooks
├── src/                   # 핵심 소스 코드
│   ├── data_processor.py  # 데이터 전처리 및 통합 로직
│   ├── feature_engineer.py# 특징 공학 로직
│   ├── model.py           # 추천 모델 정의 및 학습/평가 로직
│   └── recommender.py     # 실제 추천 파이프라인 로직
├── main.py                # 프로젝트 실행 스크립트 (학습 및 데모)
├── requirements.txt       # 프로젝트 의존성 라이브러리 목록
└── README.md              # 프로젝트 설명서
```

### 8. 개발 로드맵 (A to Z)

1.  **Phase 1: 기반 다지기 및 데이터 탐험:** 개발 환경 설정, 데이터 특성 및 분포 시각화 (EDA)
2.  **Phase 2: 데이터 정제 및 통합:** 두 데이터셋의 텍스트 정규화, 유사도 기반 병합, 최종 마스터 데이터셋 구축
3.  **Phase 3: 특징 공학 및 데이터셋 구축:** 감성(V-A) 특징 범주화, 상호작용 데이터 가중치 부여, 훈련/테스트 데이터셋 분할
4.  **Phase 4: 모델 구축 및 학습:** 하이브리드 모델 아키텍처 구현, WARP Loss 기반 학습 실행, 학습된 모델 파일 저장
5.  **Phase 5: 모델 평가 및 서비스 구현:** `Precision@k`, `AUC` 지표로 성능 평가, 2-Step 감성 입력 기반 추천 파이프라인 완성, 데모 인터페이스 제작

### 9. 기여하기 (Contributing)

이 프로젝트는 누구나 기여할 수 있는 오픈소스 프로젝트입니다. 버그 리포트, 기능 제안, 코드 개선 등 어떤 형태의 기여든 환영합니다. `Issues` 탭에 이슈를 남기거나, `Pull Request`를 보내주세요.

### 10. 라이선스 (License)

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.

```
MIT License

Copyright (c) 2025 AhnJaehyun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
