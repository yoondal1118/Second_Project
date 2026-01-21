import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# DB 연결 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '192.168.60.129'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'crawler'),
    'password': os.getenv('DB_PASSWORD', 'jbnuezen1!'),
    'db': os.getenv('DB_NAME', 'kiwi'),
    'charset': 'utf8mb4'
}

# [장르별 Aspect 필터링 설정]
# DB의 app_genre.ag_name 값에 따라 분석할 aspect_type.at_type 리스트를 정의합니다.
# 만약 여기에 정의되지 않은 장르라면 모든 Aspect를 분석합니다.
GENRE_ASPECT_MAP = {
    "엔터테인먼트": ["재생 및 화질", "앱 안정성 및 설치", "콘텐츠 및 기능", "구독 및 결제", "서비스 및 UI", "로그인 및 인증"],
    # 필요에 따라 추가 (DB의 ag_name과 일치해야 함)
}