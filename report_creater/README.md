# 📱 앱 리뷰 AI 분석 및 리포트 자동 생성기

이 프로젝트는 DB에 저장된 **ABSA(속성 기반 감성 분석) 데이터**를 활용하여, 앱 리뷰를 장르별로 분석하고 **Google Gemini AI**를 통해 전문적인 인사이트 리포트를 생성하는 자동화 시스템입니다.

생성된 리포트는 마크다운 형식으로 다시 데이터베이스(`analytics` 테이블)에 저장됩니다.

---

## 🛠️ 사전 준비 (Prerequisites)

이 프로젝트를 실행하기 위해서는 아래 항목들이 필요합니다.

*   **Python 3.8+**
*   **MySQL Database** (스키마에 맞는 데이터가 적재되어 있어야 함)
*   **Google Gemini API Key**

### 필수 라이브러리 설치
```bash
pip install pymysql pandas python-dotenv google-generativeai
```

---

## ⚙️ 환경 설정 (Configuration)

### 1. 환경 변수 파일 (`.env`) 생성
프로젝트 루트 경로에 `.env` 파일을 생성하고, **DB 접속 정보**와 **Gemini API Key**를 입력하세요.

```env
host=127.0.0.1
port=3306
user=사용자명
passwd=비밀번호
dbname=데이터베이스명

# Google Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. 장르별 분석 기준 설정 (`config.py`)
앱의 장르(Category)마다 중점적으로 분석해야 할 **Aspect(요소)**가 다릅니다.
`config.py` 파일 내 `GENRE_ASPECT_MAP` 변수에 DB의 장르명(`ag_name`)과 분석 요소(`at_type`)를 매핑해주세요.

```python
# config.py 예시

GENRE_ASPECT_MAP = {
    "엔터테인먼트": [
        "재생 및 화질", 
        "앱 안정성 및 설치", 
        "콘텐츠 및 기능", 
        "구독 및 결제", 
        "서비스 및 UI", 
        "로그인 및 인증"
    ],
    "쇼핑": [
        "배송", 
        "가격", 
        "상품 품질", 
        "고객 응대"
    ],
    # DB의 장르명과 정확히 일치해야 합니다.
}
```

---

## 🚀 사용 방법 (Usage)

1.  `main.py` 파일을 엽니다.
2.  분석하고자 하는 **앱 이름**을 `TARGET_APP` 변수에 입력합니다.

```python
# main.py 하단

if __name__ == "__main__":
    TARGET_APP = "쿠팡플레이"  # 분석할 앱 이름 입력
    main(TARGET_APP)
```

3.  코드를 실행합니다.

```bash
python main.py
```

4.  실행이 완료되면 터미널에 로그가 출력되고, DB의 `analytics` 테이블에 리포트가 저장됩니다.

---

## 📂 프로젝트 구조 및 파일 설명

### 1️⃣ `main.py` (실행 진입점)
*   프로그램의 메인 컨트롤러입니다.
*   **미분석 버전 탐색 → 데이터 조회 → 데이터 가공 → AI 리포트 생성 → DB 저장**의 전체 프로세스를 제어합니다.

### 2️⃣ `config.py` (설정 관리)
*   환경 변수 로드 및 **장르별 분석 Aspect 매핑(`GENRE_ASPECT_MAP`)**을 관리합니다.
*   분석 기준을 수정하고 싶다면 이 파일을 편집하세요.

### 3️⃣ `db_utils.py` (DB 유틸리티)
*   데이터베이스 연결 및 CRUD 작업을 담당합니다.
    *   `get_unanalyzed_versions`: 리포트가 없는 버전 탐색
    *   `fetch_reviews_by_version_ids`: 리뷰 데이터 및 평점 조회
    *   `save_report_to_analytics`: 생성된 AI 리포트 저장

### 4️⃣ `data_handler.py` (데이터 가공)
*   Raw 데이터를 분석에 용이한 형태로 통계 처리 및 가공합니다.
*   **주요 기능**:
    *   장르에 맞지 않는 불필요한 Aspect 필터링
    *   긍정/부정 비율 계산
    *   **대표 리뷰 추출** (가장 긍정적인 리뷰, 가장 부정적인 리뷰, 개선 요청이 담긴 리뷰 등)

### 5️⃣ `gemini_agent.py` (AI 에이전트)
*   가공된 데이터를 바탕으로 프롬프트를 구성하여 **Gemini API**에 전송합니다.
*   AI로부터 마크다운 형식의 전문적인 분석 보고서를 받아옵니다.

---

## 📊 결과물 예시

생성된 리포트는 DB에 저장되며, 아래와 같은 구조를 가집니다.

> **[v1.0.5 분석 리포트]**
>
> **⭐ 평균 별점**: 3.8 / 5.0
>
> **1. 종합 요약**
> 전반적으로 콘텐츠에 대한 만족도는 높으나, 로그인 오류로 인한 불만이 급증하고 있습니다.
>
> **2. 주요 이슈 (Negative)**
> *   🚨 **로그인 및 인증 (부정 65%)**
>     *   *"업데이트 이후 로그인이 계속 풀립니다."*
>
> **3. 개선 제안**
> *   로그인 세션 유지 시간 연장 및 소셜 로그인 연동 모듈 점검이 필요합니다.

```
