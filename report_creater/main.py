# main.py
import json
from config import GENRE_ASPECT_MAP  # 여기서 설정값 가져옴
from db_utils import (
    get_app_genre_info, 
    save_report_to_analytics,
    get_unanalyzed_versions,       
    fetch_reviews_by_version_ids   
)
from data_handler import process_and_filter_data
from gemini_agent import generate_ai_report
from tqdm import tqdm

def main(target_app_name):
    # 1. 앱 장르 확인
    genre_name = get_app_genre_info(target_app_name)
    print(f"📂 App: {target_app_name} | Genre: {genre_name}")

    # Config에 정의된 장르인지 확인 (로그 출력용)
    if genre_name in GENRE_ASPECT_MAP:
        print(f"   └─ 분석 설정: {GENRE_ASPECT_MAP[genre_name]}")
    else:
        print(f"   └─ ⚠️ 설정된 장르가 아님. 모든 Aspect 분석.")

    # [수정] limit=1 로 테스트
    missing_versions = get_unanalyzed_versions(target_app_name)
    
    if not missing_versions:
        print("✅ 분석할 버전이 없습니다.")
        return

    print(f"🚀 분석 대상: {len(missing_versions)}개 버전")
    target_v_idxs = [item['v_idx'] for item in missing_versions]

    # 2. 데이터 조회
    raw_df = fetch_reviews_by_version_ids(target_v_idxs)
    if raw_df is None or raw_df.empty:
        print("⚠️ 데이터 없음.")
        return

    # 3. 데이터 가공
    analyzed_data_map = process_and_filter_data(raw_df, genre_name)

    # 4. 리포트 생성 및 저장 Loop
    count = 0
    count = 0
    for v_idx, info in tqdm(analyzed_data_map.items()):
        version_name = info['version']
        aspect_stats = info['stats']
        
        # [수정 1] data_handler에서 넘어온 값들을 정확히 가져옵니다.
        avg_rating = info.get('avg_rating', 0.0)
        total_reviews = info.get('total_reviews', 0)  # 리뷰 개수 가져오기
        if total_reviews < 10 :
            continue
        
        count += 1
        print(f"\n[{count}] 🤖 '{version_name}' 분석 (⭐ {avg_rating}, 👤 {total_reviews}명)")

        if not aspect_stats:
            print("   ⚠️ Aspect 데이터 없음.")
            continue

        json_input = json.dumps(aspect_stats, ensure_ascii=False, indent=2)
        
        # [수정 2] 함수 정의 순서에 맞게 인자를 전달합니다.
        # 정의: generate_ai_report(app_name, version, genre, json_data, total_reviews, avg_rating)
        report_text = generate_ai_report(
            app_name=target_app_name,
            version=version_name,
            genre=genre_name,
            json_data=json_input,       # 순서 4: JSON 데이터
            total_reviews=total_reviews, # 순서 5: 총 리뷰 수
            avg_rating=avg_rating        # 순서 6: 평점
        )
        
        print(report_text)
        # DB 저장
        save_report_to_analytics(v_idx, report_text)
    
    print(f"\n🎉 완료.")

if __name__ == "__main__":
    TARGET_APPS = ["쿠팡플레이", "Apple TV", "Prime Video", "Wavve (웨이브)", "TVING", "왓챠", "Disney+", "Netflix(넷플릭스)"]
    for t in TARGET_APPS:
        main(t)
        