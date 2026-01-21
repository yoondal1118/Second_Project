import os
import markdown
import pdfkit  # pip install pdfkit
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from common.decorators import login_required
from common.models import User, Analytics
import json
from .ai_service import generate_chat_response
from django.db import connection
from asgiref.sync import sync_to_async

# 메인 대시보드
@login_required
def main(request):
    """
    메인 대시보드 (로그인 필수)
    """
    cursor = connection.cursor()
    user_id = request.session.get('user_id')
    user_name = request.session.get('user_name')

    # 유저 정보 확인
    cursor.execute("SELECT u_name, u_email, u_id FROM user WHERE u_idx = %s", [user_id])
    user_data = cursor.fetchone()

    if not user_data:
        cursor.close()
        return redirect('core:index')
    
    # 앱 리스트
    cursor.execute("""
        SELECT a.a_idx, a.a_developer, a.a_icon
        FROM user_app_list ual
        JOIN app a ON ual.a_idx = a.a_idx
        WHERE ual.u_idx = %s
    """, [user_id])
    user_apps = cursor.fetchall()

    # 보고서 데이터 가져오기
    cursor.execute("""
        SELECT 
            an.an_idx,
            an.an_text,
            v.v_version,
            a.a_name,
            DATE_FORMAT(MIN(r.r_date), '%%Y-%%m') as report_month,
            CONCAT(YEAR(MIN(r.r_date)), ' Q', QUARTER(MIN(r.r_date))) as report_quarter
        FROM analytics an
        JOIN version v ON an.v_idx = v.v_idx
        JOIN app a ON v.a_idx = a.a_idx
        JOIN user_app_list ual ON a.a_idx = ual.a_idx
        LEFT JOIN review r ON r.v_idx = v.v_idx
        WHERE ual.u_idx = %s
        GROUP BY an.an_idx, an.an_text, v.v_version, a.a_name
        ORDER BY 
            YEAR(MIN(r.r_date)) DESC,
            QUARTER(MIN(r.r_date)) DESC,
            CAST(SUBSTRING_INDEX(v.v_version, '.', 1) AS UNSIGNED) DESC,
            CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(v.v_version, '.', 2), '.', -1) AS UNSIGNED) DESC,
            CAST(SUBSTRING_INDEX(v.v_version, '.', -1) AS UNSIGNED) DESC
    """, [user_id])
    analytics_data = cursor.fetchall()

    # Markdown을 HTML로 변환
    reports = []
    for row in analytics_data:
        an_idx, an_text, v_version, a_name, report_month, report_quarter = row
        html_content = markdown.markdown(
            an_text or "",
            extensions=['extra', 'codehilite', 'tables', 'fenced_code']
        )
        reports.append({
            'an_idx': an_idx,
            'content': html_content,
            'version': v_version,
            'app_name': a_name,
            'month': report_month,
            'quarter': report_quarter
        })

    cursor.close()

    try:
        user = User.objects.get(u_idx=user_id)
        user_company = user_apps[0][1] if user_apps else "소속 없음"
        user_app_icon = user_apps[0][2] if user_apps else None
        
        context = {
            'user_name': user_name,
            'user_email': user.u_email,
            'user_company': user_company,
            'user_app_icon': user_app_icon,
            'reports': reports,
        }
        return render(request, 'main/main.html', context)
    
    except User.DoesNotExist:
        request.session.flush()
        return redirect('core:index')

# 보고서 내용 조회 (AJAX)
def get_report(request, report_id):
    if request.method == 'GET':
        cursor = connection.cursor()
        cursor.execute("""
            SELECT an.an_text, v.v_version, a.a_name
            FROM analytics an
            JOIN version v ON an.v_idx = v.v_idx
            JOIN app a ON v.a_idx = a.a_idx
            WHERE an.an_idx = %s
        """, [report_id])
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            an_text, v_version, a_name = row
            html_content = markdown.markdown(
                an_text or "",
                extensions=['extra', 'codehilite', 'tables', 'fenced_code']
            )
            return JsonResponse({
                'success': True,
                'content': html_content,
                'version': v_version,
                'app_name': a_name
            })
        else:
            return JsonResponse({'success': False, 'error': '보고서를 찾을 수 없습니다'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})


# PDF 다운로드 View (pdfkit 사용)
def download_report_pdf(request, report_id):
    try:
        # 1. 데이터 가져오기
        report = Analytics.objects.select_related('v_idx__a_idx').get(an_idx=report_id)
        
        app_name = report.v_idx.a_idx.a_name
        version = report.v_idx.v_version
        
        # 2. 마크다운 변환
        html_content = markdown.markdown(
            report.an_text,
            extensions=['tables', 'fenced_code', 'nl2br']
        )

        # 3. HTML & CSS 구성 (이모지 폰트 + 전체 스타일 포함)
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                /* [폰트 설정] 
                   한글: 맑은 고딕 (Malgun Gothic)
                   이모지: Segoe UI Emoji (윈도우), Noto/Apple Emoji (맥/리눅스 대비)
                */
                body {{
                    font-family: 'Malgun Gothic', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif;
                    font-size: 10pt;
                    line-height: 1.6;
                    color: #333;
                }}
                
                /* [제목 스타일] */
                h1 {{
                    font-size: 24pt;
                    color: #004085;
                    border-bottom: 2px solid #004085;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                    font-weight: bold;
                    /* 제목에도 이모지가 들어갈 수 있으므로 폰트 패밀리 명시 */
                    font-family: 'Malgun Gothic', 'Segoe UI Emoji', sans-serif;
                }}
                
                h2 {{
                    font-size: 16pt;
                    color: #333;
                    background-color: #f1f3f5;
                    padding: 8px 12px;
                    border-left: 6px solid #004085;
                    margin-top: 35px;
                    margin-bottom: 15px;
                    font-weight: bold;
                    font-family: 'Malgun Gothic', 'Segoe UI Emoji', sans-serif;
                }}

                h3 {{
                    font-size: 13pt;
                    color: #0056b3;
                    margin-top: 25px;
                    margin-bottom: 10px;
                    font-weight: bold;
                    border-bottom: 1px dotted #ccc;
                    padding-bottom: 3px;
                }}
                
                /* [테이블 스타일] */
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background-color: #004085;
                    color: white;
                    padding: 10px;
                    border: 1px solid #003060;
                    text-align: center;
                    font-weight: bold;
                }}
                td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    vertical-align: top;
                }}
                
                /* [인용구/리뷰 스타일] */
                blockquote {{
                    background: #f9f9f9;
                    border-left: 5px solid #ffc107; /* 노란색 강조선 */
                    padding: 10px 15px;
                    margin: 10px 0;
                    color: #555;
                }}
                
                /* [코드 블록 스타일] */
                pre {{
                    background: #333;
                    color: #fff;
                    padding: 10px;
                    border-radius: 5px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    font-family: Consolas, monospace; /* 코드는 고정폭 폰트 */
                }}
                
                /* [강조 스타일] */
                strong {{
                    color: #d63384; /* 핑크빛 빨강 */
                    font-weight: bold;
                }}

                /* [리스트 스타일] */
                ul, ol {{
                    margin-left: 20px;
                    margin-bottom: 15px;
                }}
                li {{
                    margin-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>{app_name} - {version}</h1>
            <div style="text-align: center; color: #888; margin-bottom: 40px; font-size: 9pt;">
                Generated by AI Analytics Report
            </div>
            {html_content}
        </body>
        </html>
        """

        # 4. wkhtmltopdf 실행 경로 설정 (Windows)
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        
        if not os.path.exists(path_wkhtmltopdf):
             return HttpResponse(f"wkhtmltopdf.exe를 찾을 수 없습니다. 경로를 확인하세요: {path_wkhtmltopdf}", status=500)

        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

        # 5. PDF 옵션 설정
        options = {
            'page-size': 'A4',
            'encoding': "UTF-8",
            'margin-top': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'margin-right': '20mm',
            'no-outline': None,
            # 이모지 렌더링 개선을 위한 스마트 쉬링크 끄기 (선택사항)
            # 'disable-smart-shrinking': None 
        }

        # 6. PDF 생성
        pdf_file = pdfkit.from_string(full_html, False, configuration=config, options=options)

        # 7. 파일 반환
        response = HttpResponse(pdf_file, content_type='application/pdf')
        filename = f"Report_{app_name}_{version}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return HttpResponse(f'PDF 생성 오류: {str(e)}', status=500)
    
# DB 접근 함수 (반드시 async 뷰 밖에서 정의)
def get_user_apps_db(user_id):
    """
    동기 방식으로 DB에 접근하여 유저의 앱 이름 리스트를 반환합니다.
    """
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT a.a_name 
            FROM user_app_list ual
            JOIN app a ON ual.a_idx = a.a_idx
            WHERE ual.u_idx = %s
        """, [user_id])
        return [row[0] for row in cursor.fetchall()]

# 비동기 채팅 뷰
async def chat_api(request):
    if request.method == "POST":
        try:
            # 1. 데이터 파싱
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            # 2. 세션 유저 확인
            @sync_to_async
            def get_session_user_id():
                return request.session.get('user_id')
            
            user_id = await get_session_user_id()

            if not user_message:
                return JsonResponse({'error': '내용을 입력해주세요.'}, status=400)
            
            if not user_id:
                return JsonResponse({'error': '로그인이 필요합니다.'}, status=401)

            # 3. DB에서 앱 리스트 가져오기 (미리 실행)
            try:
                user_app_list = await sync_to_async(get_user_apps_db)(user_id)
            except Exception as db_error:
                print(f"DB Error: {db_error}")
                user_app_list = [] 

            # 4. [핵심 수정] StreamingHttpResponse 사용
            # 주의: generate_chat_response 앞의 'await'를 제거하고 함수 호출 자체를 넘깁니다.
            # content_type='text/plain'으로 설정하여 텍스트가 오는 대로 화면에 뿌릴 수 있게 합니다.
            
            response_generator = generate_chat_response(user_message, user_app_list)
            
            return StreamingHttpResponse(
                response_generator, 
                content_type='text/plain; charset=utf-8'
            )
            
        except Exception as e:
            import traceback
            print(f"Chat Error: {e}")
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'error': '서버 오류가 발생했습니다.'}, status=500)
    
    return JsonResponse({'error': '잘못된 요청입니다.'}, status=405)