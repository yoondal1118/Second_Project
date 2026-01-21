# Frontend / Django Setup
프론트엔드(Django) 관련 파일 구조, 초기 세팅 및 보안 설정 정리

```bash
# 파일 로드 후 이 명령어 입력.
cd mysite
```

---

## 1. 환경 설정 및 설치 (초기 세팅)
미니콘다 활성화 상태에서 필수 라이브러리를 설치.

```bash
# Django 설치 (최신 버전)
pip install django

# mysql 설치 (mysqlclient C++ 컴파일 오류 방지를 위해 pymysql 사용)
pip install pymysql

# 환경변수(.env) 관리 도구 설치
pip install python-dotenv

# 마크다운 렌더링 라이브러리
pip install markdown

# PDF로 저장하는 라이브러리 (https://wkhtmltopdf.org/downloads.html > Windows > Installer (Vista or later) > 64-bit 다운로드)
pip install pdfkit

# MYSQL 접근에 필요한 라이브러리
pip install cryptography

# 개발 서버를 ASGI 모드로 바꾸는 라이브러리
pip install daphne
```

<br>

## 2. 프로젝트 및 앱 생성

### 2-1. 프로젝트 생성
`mysite`라는 이름의 메인 프로젝트 컨테이너를 생성.
```bash
django-admin startproject mysite
```

### 2-2. 프로젝트 폴더로 이동 (★필수)
모든 `manage.py` 명령어는 이 폴더 안에서 실행.
```bash
cd mysite
```

### 2-3. 앱 생성
기능 단위(예: core, mypage, main 등)의 앱을 생성.
```bash
# 예: core 앱 생성
python manage.py startapp core
```
> **Note:** 앱을 생성한 후 반드시 `mysite/settings.py`의 `INSTALLED_APPS` 리스트에 앱 이름 추가.

```python
# mysite/settings.py
INSTALLED_APPS = [
    # ... 기존 앱들 ...
    'core', # 새로 생성한 앱 추가
]
```

<br>

## 3. 보안 및 DB 설정 (.env)
민감한 정보(비밀번호, API Key)는 소스코드에서 분리하여 관리.

### 3-1. .env 파일 생성
프로젝트 루트(`manage.py`와 같은 위치)에 `.env` 파일을 생성하고 아래 내용을 작성.   
**(★주의: `.env` 파일은 반드시 `.gitignore`에 추가하여 깃허브에 올라가지 않도록 설정)**

```ini
# .env 파일 내용 (예시)

# Django Secret Key
SECRET_KEY=django-insecure-xxxxx(settings.py에_있는_키_복사)

# Database Connection Info
DB_ENGINE=django.db.backends.mysql
DB_NAME=[DB_NAME]
DB_USER=[DB_USER_NAME]
DB_PASSWORD=[DB_USER_PASSWORD]
DB_HOST=[DB_IP_ADDRESS]
DB_PORT=[DB_PORT]

# API Keys (예시)
OPENAI_API_KEY=sk-xxxx...
KAKAO_MAP_API_KEY=abcdef...
```

### 3-2. settings.py 수정
`mysite/settings.py`에서 `.env` 파일을 읽어오도록 코드 수정.

```python
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# SECRET_KEY 적용
SECRET_KEY = os.getenv('SECRET_KEY')

# DATABASES 설정 적용
DATABASES = {
    'default': {
        'ENGINE': os.getenv('DB_ENGINE'),
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': os.getenv('DB_PORT'),
        'OPTIONS': {
            'charset': 'utf8mb4',
        },
    }
}

# API Key 설정 (Django 전역 변수로 등록)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KAKAO_MAP_API_KEY = os.getenv('KAKAO_MAP_API_KEY')
```

### 3-3. PyMySQL 연동 설정 (mysqlclient 오류 해결)
Django는 기본적으로 C로 컴파일된 `mysqlclient`를 요구하지만, Windows 환경에서는 설치 오류가 잦음. 이를 순수 Python 패키지인 `PyMySQL`로 대체하기 위한 설정.

`settings.py`와 같은 폴더에 있는 `mysite/__init__.py` 파일을 열고 아래 코드를 추가.

```python
# mysite/__init__.py

import pymysql

# Django의 MySQL 버전 체크를 통과하기 위한 속임수 (Bypass)
pymysql.version_info = (2, 2, 7, "final", 0) 

# pymysql을 MySQLdb처럼 작동하게 만듦
pymysql.install_as_MySQLdb()
```

<br>

## 4. 데이터베이스(DB) 모델링 및 적용

### 4-1. 기존 MySQL DB → Django 모델 변환 (리버스 엔지니어링)
DB 구조를 읽어 `existing_models.py` 파일로 저장.
```bash
python manage.py inspectdb > existing_models.py
```

### 4-2. 모델 변경사항 감지
모델 파일(`models.py`)의 변경 사항을 추적하여 마이그레이션 파일을 생성.
```bash
python manage.py makemigrations
```

### 4-3. DB 적용 (Migrate)
Django의 기본 테이블(admin, auth 등)과 변경된 모델 내용을 실제 데이터베이스에 적용.
```bash
python manage.py migrate
```

<br>

## 5. 관리자 생성 및 서버 실행

### 5-1. 최고 관리자(Superuser) 생성
관리자 페이지(`admin/`) 접속을 위한 계정을 생성.
```bash
python manage.py createsuperuser
```

### 5-2. 개발 서버 실행
웹사이트를 로컬 환경에서 실행. (기본 주소: http://127.0.0.1:8000/)
```bash
python manage.py runserver
```

<br>

---

## 6. API Key 사용 가이드 (Backend & Frontend 통합)
`.env`에 저장된 키를 상황(서버 로직 vs HTML 화면)에 따라 사용하는 방법.

### 6-1. Case A: 백엔드 로직에서 사용 (Python)
서버 내부에서 AI를 호출하거나 DB에 접근할 때 사용. (사용자에게 키가 노출되지 않음)

```python
# views.py (또는 services.py)
from django.conf import settings # settings를 불러옴
import openai

def chat_with_ai(request):
    # settings.py에 등록된 변수를 바로 사용
    openai.api_key = settings.OPENAI_API_KEY
    
    # ... AI 로직 수행 ...
```

### 6-2. Case B: 프론트엔드 화면에서 사용 (HTML/JS)
지도 API(Kakao, Google)처럼 브라우저에서 키가 필요한 경우.
**흐름:** `.env` → `settings.py` → `views.py` → `HTML`

**1. views.py에서 전달**
```python
# core/views.py
from django.shortcuts import render
from django.conf import settings

def map_view(request):
    context = {
        # settings에 있는 키를 템플릿으로 넘김
        'map_api_key': settings.KAKAO_MAP_API_KEY, 
    }
    return render(request, 'core/map.html', context)
```

**2. HTML에서 사용**
```html
<!-- core/templates/core/map.html -->
<!-- Django 템플릿 문법 {{ }} 를 사용하여 키 삽입 -->
<script type="text/javascript" src="//dapi.kakao.com/.../sdk.js?appkey={{ map_api_key }}"></script>
```

> **Security Tip:** 프론트엔드 키는 브라우저 소스 보기를 통해 노출될 수 있으므로, 해당 API 제공사(Kakao, Google 등) 콘솔에서 **'허용 도메인(Web Domain)'** 설정을 통해 내 사이트(`http://127.0.0.1:8000` 등)에서만 동작하도록 제한해야 함.

<br>

---

## 7. 프로젝트 주요 파일 구조 설명

| 위치 | 파일/폴더명 | 역할 및 설명 |
|---|---|---|
| **Root** | `manage.py` | 프로젝트 관리 명령어(실행, DB설정 등)를 수행하는 도구 |
| | `.env` | **(보안)** DB 비밀번호, API Key 등 민감 정보 저장소 |
| **Config** | `mysite/settings.py` | 프로젝트 전체 설정 (DB 정보, 앱 등록, 시간대, API Key 로드) |
| | `mysite/__init__.py`| 패키지 초기화 파일 **(PyMySQL 연동 설정 포함)** |
| | `mysite/urls.py` | 사이트의 메인 주소(URL) 진입점 (길 안내) |
| **App** | `core/models.py` | **데이터베이스 설계도**. 테이블 구조를 정의하는 곳 |
| | `core/views.py` | **로직 담당**. 데이터를 가공하고 API Key 등을 템플릿으로 전달 |
| | `core/urls.py` | 앱 내부의 상세 주소 관리 |
| | `core/templates/` | 사용자에게 보여질 화면(HTML) 파일 저장소 |
