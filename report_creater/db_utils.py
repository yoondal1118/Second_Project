import pymysql
import pandas as pd
from config import DB_CONFIG

def get_db_connection():
    """DB 연결 객체 생성"""
    return pymysql.connect(
        **DB_CONFIG,
        cursorclass=pymysql.cursors.DictCursor
    )

def get_app_genre_info(app_name):
    """
    [Modified] 앱 이름으로 장르명을 조회합니다.
    App 테이블의 ag_idx를 통해 app_genre 테이블과 JOIN합니다.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 수정된 부분: JOIN 쿼리 사용
            sql = """
                SELECT ag.ag_name 
                FROM app a
                JOIN app_genre ag ON a.ag_idx = ag.ag_idx
                WHERE a.a_name = %s
            """
            cursor.execute(sql, (app_name,))
            result = cursor.fetchone()
            return result['ag_name'] if result else "General"
    finally:
        conn.close()

def get_unanalyzed_versions(app_name, limit=None):
    """
    analytics 테이블에 리포트가 존재하지 않는 버전들의 ID(v_idx)를 조회합니다.
    limit 파라미터로 개수를 제한할 수 있습니다.
    """
    conn = get_db_connection()
    
    sql = """
        SELECT v.v_idx, v.v_version
        FROM version v
        JOIN app a ON v.a_idx = a.a_idx
        LEFT JOIN analytics an ON v.v_idx = an.v_idx
        WHERE a.a_name = %s
          AND an.an_idx IS NULL
        ORDER BY v.v_idx DESC
    """
    
    if limit:
        sql += f" LIMIT {int(limit)}"

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (app_name,))
            rows = cursor.fetchall()
            return rows 
    finally:
        conn.close()

def fetch_reviews_by_version_ids(target_v_idxs):
    """
    특정 버전 ID 리스트에 해당하는 리뷰 데이터를 가져옵니다.
    [중요] 별점 분석을 위해 r.r_score를 함께 조회합니다.
    """
    if not target_v_idxs:
        return None

    conn = get_db_connection()
    
    format_strings = ','.join(['%s'] * len(target_v_idxs))
    
    sql = f"""
        SELECT 
            v.v_idx,
            v.v_version,
            r.r_idx,
            r.r_score,       -- [추가됨] 평균 평점 계산을 위해 필요
            rl.rl_line      AS original_segment,
            at.at_type      AS aspect,
            et.et_type      AS sentiment
        FROM version v
        JOIN review r ON v.v_idx = r.v_idx
        JOIN review_line rl ON r.r_idx = rl.r_idx
        JOIN analysis an ON rl.rl_idx = an.rl_idx
        JOIN aspect_type at ON an.at_idx = at.at_idx
        JOIN emotion_type et ON an.et_idx = et.et_idx
        WHERE v.v_idx IN ({format_strings})
        ORDER BY v.v_idx DESC
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, tuple(target_v_idxs))
            rows = cursor.fetchall()
            
        if not rows:
            return None
        
        return pd.DataFrame(rows)
    finally:
        conn.close()

def save_report_to_analytics(v_idx, report_text):
    """
    생성된 보고서를 analytics 테이블에 저장합니다.
    """
    conn = get_db_connection()
    
    sql = """
        INSERT INTO analytics (v_idx, an_text)
        VALUES (%s, %s)
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (v_idx, report_text))
        conn.commit()
        print(f"✅ [DB] 리포트 저장 완료 (v_idx: {v_idx})")
    except Exception as e:
        print(f"❌ [DB] 리포트 저장 실패: {e}")
    finally:
        conn.close()
