# KSSDS Review Sentence Splitter

이 프로젝트는 MySQL 데이터베이스에 저장된 한국어 리뷰 데이터를 가져와, 텍스트 정규화(Normalization) 과정을 거친 후 **KSSDS(Korean Sentence Splitter for Dialect and Spoken)** 라이브러리를 사용하여 문장 단위로 분리하고, 이를 별도의 테이블에 저장하는 Python 스크립트입니다.

## 📌 주요 기능

1.  **데이터베이스 연동**: MySQL DB에서 아직 처리되지 않은 리뷰 데이터를 자동으로 선별하여 가져옵니다.
2.  **텍스트 정규화**: `soynlp`를 사용하여 반복되는 문자(예: "ㅋㅋㅋㅋ" → "ㅋㅋ")를 정규화합니다.
3.  **문장 분리**: 대화체 및 방언에 강한 `KSSDS` 모델을 사용하여 리뷰를 문장 단위로 분리합니다.
4.  **데이터 필터링**: 길이가 5글자 미만인 리뷰는 처리를 생략하여 데이터 품질을 유지합니다.
5.  **오류 처리 및 로깅**: 개별 리뷰 처리 중 오류가 발생하더라도 멈추지 않고 다음 리뷰를 진행하며, 실패한 리뷰 ID를 리포트합니다.

## 🛠️ 설치 및 환경 설정

### 1. 필수 요구 사항 (Prerequisites)
*   Python 3.10
*   MySQL Database

### 2. 라이브러리 설치
아래 명령어를 통해 필요한 패키지를 설치하십시오. 호환성을 위해 `transformers`는 특정 버전을 사용하는 것이 권장됩니다.

```bash
# 가상 환경 생성 및 활성화 (권장)
conda create -n kssds_env python=3.10
conda activate kssds_env

# 의존성 패키지 설치
pip install python-dotenv
pip install pymysql
pip install soynlp
pip install cryptography
pip install KSSDS

# transformers 버전 호환성 맞추기
pip uninstall transformers -y
pip install transformers==4.30.2
```

### 3. 환경 변수 설정 (.env)
프로젝트 루트 경로에 `.env` 파일을 생성하고, 데이터베이스 접속 정보를 입력하십시오.

```env
host=127.0.0.1
port=3306
user=사용자명
passwd=비밀번호
dbname=데이터베이스명
```

## 🗄️ 데이터베이스 스키마

이 스크립트는 다음 두 테이블 구조를 가정하고 작동합니다.

### 1. `review` (원본 리뷰 테이블)
| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `r_idx` | INT (PK) | 리뷰 고유 ID |
| `r_content` | TEXT | 리뷰 원문 내용 |

### 2. `review_line` (문장 분리 결과 저장 테이블)
| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `r_idx` | INT | 원본 리뷰 ID (FK) |
| `rl_line` | TEXT | 분리된 문장 |

**테이블 생성 예시 SQL:**
```sql
CREATE TABLE review_line (
    rl_idx INT AUTO_INCREMENT PRIMARY KEY,
    r_idx INT NOT NULL,
    rl_line TEXT,
    INDEX idx_review_r_idx (r_idx)
);
```

## 🚀 사용 방법

환경 설정이 완료되었다면 아래 명령어로 스크립트를 실행합니다.

```bash
python kssds_line_splitter.py
```

## ⚙️ 작동 로직 상세

1.  **대상 조회**: `review` 테이블에는 존재하지만 `review_line` 테이블에는 없는 `r_idx`를 조회합니다 (`LEFT JOIN` 활용).
2.  **반복 정규화**: `repeat_normalize`를 사용해 불필요하게 반복되는 문자를 최대 2회로 축소합니다.
3.  **조건 검사**: 정규화된 텍스트 길이가 5글자 미만일 경우 처리를 건너뜁니다.
4.  **문장 분리**: KSSDS 모델을 통해 문장 리스트(`segments`)로 변환합니다.
5.  **DB 저장**: 분리된 문장들을 `review_line` 테이블에 `INSERT` 합니다.
6.  **결과 요약**: 실행이 끝나면 성공/실패 횟수와 실패한 리뷰의 ID 목록을 출력합니다.

## ⚠️ 주의사항

*   **초기 로딩**: KSSDS 모델은 초기화 시 시간이 다소 소요될 수 있습니다.
*   **메모리 관리**: 대량의 데이터를 처리할 경우를 대비해 스크립트 내부 로직은 `LIMIT` 없이 전체 대상을 가져오게 되어있으나, 필요시 SQL 쿼리에 `LIMIT`을 추가하여 배치 처리를 할 수 있습니다. (현재 코드는 `fetchall()`을 사용하므로 데이터 양에 주의하세요).
*   **오류 처리**: 특정 리뷰에서 `KSSDS` 분리 실패 시 해당 리뷰는 스킵되며, 프로그램 종료 시 실패한 `r_idx`가 출력되므로 추후 확인이 가능합니다.

