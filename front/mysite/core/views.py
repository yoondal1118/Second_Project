import re
import json
from django.db import connection
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth.hashers import make_password, check_password

# common 앱에서 import
from common.models import User, App, UserAppList

@ensure_csrf_cookie
def index(request):
    """홈페이지"""
    # 1. 세션에서 user_id를 가져옵니다.
    user_id = request.session.get('user_id')
    
    # 2. 이 부분이 중요합니다! 가져온 user_id가 있는지 확인해야 합니다.
    # if not request.session.get(user_id):  <-- 기존의 잘못된 코드
    if not user_id: 
        return render(request, 'core/index.html')
    
    # ... (이하 동일하지만 안전하게 가기 위해 아래 코드로 교체 권장)
    try:
        cursor = connection.cursor()
        # 유저 정보 조회
        cursor.execute("SELECT u_name, u_email, u_id FROM user WHERE u_idx = %s", [user_id])
        user_data = cursor.fetchone()

        if not user_data:
            request.session.flush()
            return redirect('core:index')

        # 앱 리스트 조회
        cursor.execute("""
            SELECT a.a_idx, a.a_developer, a.a_icon
            FROM user_app_list ual
            JOIN app a ON ual.a_idx = a.a_idx
            WHERE ual.u_idx = %s
        """, [user_id])
        user_apps = cursor.fetchall()

        # 유저 객체 정보 (템플릿 전달용)
        user_company = user_apps[0][1] if user_apps else "소속 없음"
        user_app_icon = user_apps[0][2] if user_apps else None
        
        context = {
            'user_name': user_data[0], # 세션 이름 대신 DB 이름을 써도 됨
            'user_email': user_data[1],
            'user_company': user_company,
            'user_app_icon': user_app_icon,
        }
        return render(request, 'core/index.html', context)
    except Exception as e:
        print(f"Error: {e}")
        return render(request, 'core/index.html')
    finally:
        cursor.close()

def check_id(request):
    """아이디 중복 체크"""
    if request.method == 'POST':
        data = json.loads(request.body)
        user_id = data.get('user_id', '')
        
        # 아이디 형식 체크 (영문3글자 + 숫자1개 이상, 최소 4글자)
        pattern = r'^(?=.*[a-zA-Z]{3,})(?=.*\d)[a-zA-Z\d]{4,}$'
        if not re.match(pattern, user_id):
            return JsonResponse({
                'available': False, 
                'message': '아이디는 영문 3글자 이상 + 숫자 1개 이상 포함해야 합니다.'
            })
        
        # DB에서 중복 체크
        exists = User.objects.filter(u_id=user_id).exists()
        
        if exists:
            return JsonResponse({
                'available': False,
                'message': '이미 사용중인 아이디입니다.'
            })
        else:
            return JsonResponse({
                'available': True,
                'message': '사용 가능한 아이디입니다.'
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def signup(request):
    """회원가입"""
    if request.method == 'POST':
        data = json.loads(request.body)
        
        name = data.get('name', '')
        email = data.get('email', '')
        user_id = data.get('user_id', '')
        password = data.get('password', '')
        
        # 이메일에서 회사 도메인 추출
        email_domain = email.split('@')[1] if '@' in email else ''
        
        # 디버깅: 어떤 앱이 검색되는지 확인
        company_apps = App.objects.filter(
            a_developer_email__icontains=email_domain
        )
        
        print(f"이메일: {email}")
        print(f"도메인: {email_domain}")
        print(f"찾은 앱 개수: {company_apps.count()}")
        for app in company_apps:
            print(f"- {app.a_name} (idx: {app.a_idx})")
        
        # 앱 인덱스만 추출
        app_idx_list = list(company_apps.values_list('a_idx', flat=True))
        
        # 비밀번호 암호화
        hashed_password = make_password(password)
        
        try:
            # 유저 생성
            new_user = User.objects.create(
                u_name=name,
                u_id=user_id,
                u_pw=hashed_password,
                u_email=email,
                u_admin=False
            )
            
            # UserAppList에 저장
            saved_count = 0
            for app_idx in app_idx_list:
                UserAppList.objects.create(
                    u_idx=new_user,
                    a_idx_id=app_idx
                )
                saved_count += 1
                print(f"✅ user_app_list 저장 완료: user={new_user.u_idx}, app={app_idx}")
            
            return JsonResponse({
                'success': True,
                'message': '회원가입이 완료되었습니다!',
                'company_apps_count': saved_count
            })
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            return JsonResponse({
                'success': False,
                'message': f'회원가입 중 오류가 발생했습니다: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def login_view(request):
    """로그인"""
    if request.method == 'POST':
        data = json.loads(request.body)
        
        user_id = data.get('user_id', '')
        password = data.get('password', '')
        
        try:
            # 유저 찾기
            user = User.objects.get(u_id=user_id)
            
            # 비밀번호 체크
            if check_password(password, user.u_pw):
                # 세션에 유저 정보 저장
                request.session['user_id'] = user.u_idx
                request.session['user_name'] = user.u_name
                
                return JsonResponse({
                    'success': True,
                    'message': '로그인 성공!',
                    'redirect': '/main/'  # 대시보드로 이동
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': '비밀번호가 일치하지 않습니다.'
                })
                
        except User.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': '존재하지 않는 아이디입니다.'
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def logout_view(request):
    """로그아웃"""
    if request.method == 'POST':
        # 세션 전부 삭제
        request.session.flush()
        
        return JsonResponse({
            'success': True,
            'message': '로그아웃 되었습니다'
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)