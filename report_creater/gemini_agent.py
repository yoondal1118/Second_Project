import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def generate_ai_report(app_name, version, genre, json_data, total_reviews, avg_rating=0.0):
    """
    [Updated] avg_rating 인자 추가됨
    """
    prompt = f"""
    역할: 너는 모바일 애플리케이션의 사용자 경험(UX) 및 품질 관리(QA)를 담당하는 **수석 데이터 분석가이자 PM**이다.
제공되는 '버전별 상세 분석 리포트' 데이터를 바탕으로, 개발팀과 경영진이 즉시 실행할 수 있는 수준의 **[심층 분석 전략 보고서]**를 작성하라.
    대상 앱: {app_name} (장르: {genre})
    버전: {version}
    ⭐ 평균 별점: {avg_rating} / 5.0
    
    아래 JSON 데이터는 해당 앱 버전의 사용자 리뷰를 Aspect(요소)별로 분석한 결과입니다.
    이를 바탕으로 **마크다운(Markdown)** 형식의 보고서를 작성해주세요.
    
    [데이터]
    {json_data}
    
    ### 작성 지침 (Guidelines)
1. **톤앤매너**: 전문적이고 객관적이며, 실행 중심적(Action-oriented)인 어조를 사용한다.
2. **구조화**: 데이터에 존재하는 **언급량이 높은 모든 주요 Aspect**를 다뤄야 한다.
3. **인사이트 도출**: 단순히 리뷰를 나열하지 말고, 개발자가 이해할 수 있는 **기술적 원인(Root Cause)**과 기획자가 참고할 **개선 방향(Action Plan)**을 추론하여 작성한다.
4. **우선순위**: 부정 비율이 높고 언급량이 많은 항목을 '긴급 이슈'로, 긍정 반응이 높은 항목을 '핵심 강점'으로 분류한다.

---

## Output Format (Markdown Template)
반드시 아래 형식을 유지하며 내용을 작성하라.

# 📱 [어플리케이션] 버전별 심층 분석 보고서

## 1. 📑 보고서 개요
| 항목 | 내용 |
| :--- | :--- |
| **분석 대상 버전** | {version} |
| **사용자 평점** | {avg_rating} |
| **분석 표본 수** | {total_reviews} 개의 유효 리뷰 |

---

## 2. 📊 종합 요약 (Executive Summary)
### 2.1 총평
(이 버전의 전반적인 유저 반응, 전 버전 대비 분위기, 주요 이슈를 3~4줄로 요약)

### 2.2 핵심 개선 방향 (Top Priority)
> **"가장 시급하게 해결해야 할 과제"**
- **Critical Issue**: (가장 언급량이 많거나 치명적인 문제 1~2개 요약)
- **Strategic Direction**: (이를 해결하기 위한 기술적/기획적 총평)

---

## 3. 🚨 상세 이슈 분석 및 가이드라인 (Deep Dive)
*(입력 데이터의 Aspect 중 **부정 비율이 30% 이상**이거나 **언급량이 높은** 항목들에 대해 아래 양식을 반복 작성)*

### 3.1 Aspect 이름 1 (언급량: Num, 부정 비율: Percent%)
**💬 대표 유저 보이스 (VOC)**
> 💥 **Problem**: *"데이터의 [단점] 리뷰 내용"*
> 🔧 **Request**: *"데이터의 [개선] 리뷰 내용"*

**🕵️ 원인 분석 (Root Cause)**
- (리뷰 문맥을 통해 추정한 문제의 원인. 예: 특정 기기에서의 크래시, 인증 서버 타임아웃, UI 접근성 부족 등)
- (개발적 관점 또는 UX 설계 관점에서 구체적으로 서술)

**🔧 개선 가이드라인 (Action Plan)**
1. **Immediate (즉시 조치)**: (버그 픽스, 서버 증설 등 당장 해야 할 일)
2. **Long-term (장기 개선)**: (프로세스 개선, UI 개편 등 근본적 해결책)

*(...데이터에 존재하는 나머지 모든 Aspect들도 위와 동일한 포맷으로 3.2, 3.3... 계속 이어서 작성...)*

---

## 4. 🏆 강점 분석 및 확장 전략 (Strengths & Opportunities)
*(입력 데이터 중 긍정 반응이 돋보이는 Aspect에 대해 작성)*

### 4.1 가장 긍정적인 Aspect 이름
**💬 대표 유저 보이스 (VOC)**
> 🏆 **Praise**: *"데이터의 [장점] 리뷰 내용"*

**🚀 가치 제안 (Value Proposition)**
- (유저들이 이 기능을 왜 좋아하는지, 어떤 가치를 느끼는지 분석)

**📈 확장 전략 (Scale-up Strategy)**
- (이 강점을 활용한 마케팅 포인트 또는 기능 고도화 아이디어)

---

## 5. 📝 결론 및 제언 (Conclusion)
- **종합 의견**: (이번 버전의 성과와 한계를 정리)
- **차기 버전 목표**: (다음 업데이트(Next_Version)에서 집중해야 할 KPI 및 목표 제안)
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        print(f"🤖 [AI] Gemini에게 보고서 작성 요청 중... ({version})")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ [AI] 생성 실패: {e}")
        return "보고서 생성 실패 (AI Error)"