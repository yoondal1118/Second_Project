import re
import pandas as pd
from config import GENRE_ASPECT_MAP

def calculate_review_score(text, mode='general'):
    """리뷰 텍스트 점수 계산 (변경 없음)"""
    if not isinstance(text, str): return 0
    text = text.strip()
    score = 0
    
    if len(text) < 5: return 0
    score += min(len(text), 200) * 0.1

    if mode == 'request':
        request_patterns = [r'주세요', r'좋겠어요', r'바랍니다', r'부탁', r'기원', r'제발', r'추가 좀', r'수정 좀', r'내놔', r'해줘']
        matched = False
        for pat in request_patterns:
            if re.search(pat, text):
                score += 20
                matched = True
        if not matched: score -= 10
    else:
        if re.search(r'[?!ㅠㅋㅎㅡ🤬😡😢👍]', text): score += 10
        if re.search(r'[0-9]', text): score += 5
        keywords = ['오류', '버그', '최악', '좋음', '만족', '환불', '접속', '무한', '로딩', '결제']
        for k in keywords:
            if k in text: score += 5
    return score

def process_and_filter_data(df, genre_name):
    """
    DataFrame을 받아서 장르(genre_name)에 맞는 Aspect만 남기고 통계를 냅니다.
    반환값: { v_idx: { version_name: "1.0", avg_rating: 4.5, data: [...] }, ... }
    """
    processed_result = {}
    
    # 점수 계산 (기존 로직)
    df['general_score'] = df['original_segment'].apply(lambda x: calculate_review_score(x, 'general'))
    df['request_score'] = df['original_segment'].apply(lambda x: calculate_review_score(x, 'request'))
    
    # 버전별 그룹핑
    for (v_idx, ver_name), ver_df in df.groupby(['v_idx', 'v_version']):
        total_sentences = len(ver_df)
        
        # [NEW] 1. 버전별 평균 별점 및 참여자 수 계산
        # 문장 단위(ver_df)가 아니라 리뷰 단위로 중복 제거 후 계산해야 정확함
        unique_reviews = ver_df.drop_duplicates(subset=['r_idx'])
        avg_rating = round(unique_reviews['r_score'].mean(), 2) if not unique_reviews.empty else 0.0
        user_count = len(unique_reviews)

        aspect_list = []
        
        # 2. Config에서 해당 장르의 허용 Aspect 리스트 가져오기
        allowed_aspects = GENRE_ASPECT_MAP.get(genre_name)
        
        for aspect, asp_df in ver_df.groupby('aspect'):
            if aspect == '의견없음': continue
            if allowed_aspects and aspect not in allowed_aspects:
                continue 
            
            mention_count = len(asp_df)
            share_ratio = round((mention_count / total_sentences) * 100, 1) if total_sentences > 0 else 0
            
            neg_cnt = len(asp_df[asp_df['sentiment'] == '부정'])
            neg_ratio = round((neg_cnt / mention_count) * 100, 1)
            
            # 대표 리뷰 추출
            best_pros = asp_df[asp_df['sentiment'] == '긍정'].sort_values('general_score', ascending=False)
            best_cons = asp_df[asp_df['sentiment'] == '부정'].sort_values('general_score', ascending=False)
            best_improv = asp_df[asp_df['request_score'] >= 15].sort_values('request_score', ascending=False)

            aspect_list.append({
                "aspect": aspect,
                "count": mention_count,
                "share_percent": share_ratio,
                "negative_percent": neg_ratio,
                "reviews": {
                    "good": best_pros.iloc[0]['original_segment'] if not best_pros.empty else None,
                    "bad": best_cons.iloc[0]['original_segment'] if not best_cons.empty else None,
                    "request": best_improv.iloc[0]['original_segment'] if not best_improv.empty else None
                }
            })
        
        aspect_list.sort(key=lambda x: (x['negative_percent'], x['count']), reverse=True)
        
        # [NEW] 결과 구조에 avg_rating 추가
        processed_result[v_idx] = {
            "version": ver_name,
            "avg_rating": avg_rating,     # 평균 별점
            "total_reviews": user_count,  # 리뷰 참여 수
            "stats": aspect_list
        }
        
    return processed_result