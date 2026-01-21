import os
import asyncio
from django.conf import settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

# .env 로드
load_dotenv()

PERSIST_DIRECTORY = "./RAG/chroma_advanced_db"

print(f"📂 [System] AI 서비스 초기화 중... (DB 경로: {PERSIST_DIRECTORY})")

# 전역 변수
retriever = None 
llm = None  # 단일 모델 변수

try:
    embeddings = OllamaEmbeddings(model="bge-m3")
    reranker_model = HuggingFaceCrossEncoder(model_name="dragonkue/bge-reranker-v2-m3-ko")

    # [수정 1] 복잡한 모델 dict 제거하고 메인 모델 하나만 정의
    # Llama-3.3-70b는 컨텍스트 이해력이 좋아 바로 사용해도 무방합니다.
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, streaming=True) 

    # 벡터 DB 로드
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 검색 파이프라인 (기존 로직 유지 - 검색 품질은 중요하므로)
    def get_search_pipeline(vector_store):
        try:
            all_docs_data = vector_store.get()
            texts = all_docs_data["documents"]
            metadatas = all_docs_data["metadatas"]
            
            if not texts:
                print("⚠️ [Warning] 벡터 DB에 문서가 없습니다.")
                return None
            
            doc_objects = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
            bm25_retriever = BM25Retriever.from_documents(doc_objects)
            bm25_retriever.k = 15
            
            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
            
            compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
            final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble
            )
            return final_retriever
        except Exception as e:
            print(f"❌ Pipeline Init Error: {e}")
            return None

    retriever = get_search_pipeline(vector_store)
    
    if retriever:
        print("✅ [System] AI 모델 및 DB 로드 완료")
    else:
        print("⚠️ [System] 리트리버 생성 실패")

except Exception as e:
    print(f"❌ [Critical] AI 서비스 초기화 실패: {e}")
    retriever = None


async def generate_chat_response(query, valid_apps):
    if not retriever or not llm:
        yield "죄송합니다. 현재 AI 서비스를 사용할 수 없습니다."
        return

    # 대소문자나 공백 차이로 인한 오류 방지를 위해 전처리
    valid_apps_clean = [app.strip().replace(" ", "").lower() for app in valid_apps]
    valid_apps_str = ", ".join([f"'{app}'" for app in valid_apps]) or "(등록된 앱 없음)"

    # 1. 검색 수행
    try:
        retrieved_docs = await asyncio.to_thread(retriever.invoke, query)
        print(f"🔍 [Debug] 검색된 문서 개수: {len(retrieved_docs)}개") # 디버깅용
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        yield "문서 검색 중 오류가 발생했습니다."
        return

    # 2. 문맥 구성 및 필터링
    context_list = []
    used_docs = []

    if retrieved_docs:
        for d in retrieved_docs:
            # 메타데이터에서 앱 이름 가져오기 (없으면 '알수없음')
            raw_app_name = d.metadata.get('app_name', '알수없음')
            clean_app_name = raw_app_name.strip().replace(" ", "").lower()
            
            # [수정 포인트] 정확히 일치하지 않아도 포함되어 있다면 허용 (부분 일치 로직)
            is_valid = any(app in clean_app_name or clean_app_name in app for app in valid_apps_clean)
            
            if is_valid:
                version_info = d.metadata.get('version', '알수없음')
                source_info = f"[[분석 대상 앱: {raw_app_name}, 버전: {version_info}]]"
                summary_text = d.metadata.get('summary', '')
                full_text = f"{source_info}\n요약: {summary_text}\n상세내용: {d.page_content}"
                
                context_list.append(full_text)
                used_docs.append(d)
            else:
                print(f"⚠️ [Filtered] 허용되지 않은 앱 제외됨: {raw_app_name}")

    # 문맥이 없는 경우 (검색은 됐으나 필터에서 다 걸러진 경우 포함)
    if not context_list:
        yield (
            f"검색 결과, 현재 등록된 앱({valid_apps_str}) 목록에서 **'{query}'**와 관련된 상세 보고서를 찾지 못했습니다.\n"
            "앱 이름이 정확한지, 혹은 해당 앱에 대한 분석 데이터가 업로드되었는지 확인해주세요."
        )
        return

    # --- 이후 프롬프트 및 스트리밍 로직은 동일하게 유지 ---
    context = "\n\n".join(context_list)

    # 3. 프롬프트 구성 (동일함)
    rag_prompt_template = f"""
    당신은 유능한 앱 리뷰 데이터 분석가입니다. 아래 제공된 [검색된 보고서 문맥]을 바탕으로 사용자의 질문에 친절하게 답변하세요.

    [사용자의 등록된 앱 목록]
    {valid_apps_str}

    [검색된 보고서 문맥]
    {{context}}

    [사용자 질문]
    {{query}}

    [🚨 답변 작성 가이드라인]
    1. **앱 이름 및 버전 유연성**: 
       - 질문에 포함된 앱 이름이 [등록된 앱 목록]에 포함된다면 정상적으로 분석하세요.
       - 질문한 특정 '빌드 번호'가 문맥에 없더라도, 문맥상 가장 유사하거나 최신 버전을 기반으로 답변하고, "정확한 빌드 번호는 없지만 유사 버전에 따르면..."이라고 언급하세요.
    
    2. **데이터 기반 답변**:
       - 문맥에 없는 내용은 지어내지 말고 "해당 내용은 보고서에서 확인할 수 없습니다"라고 솔직히 말하세요.
       - "NO_DATA", "NOT_REGISTERED" 같은 시스템 코드를 출력하지 말고 자연스러운 문장으로 설명하세요.

    3. **일상 대화 처리**:
       - "안녕", "고마워" 같은 인사말에는 분석가 페르소나에 맞춰 정중하게 인사하세요.

    4. **스타일**:
       - 전문적이면서도 이해하기 쉬운 구어체(~해요)를 사용하세요.
    """
    
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    # 체인 생성
    chain = prompt | llm | StrOutputParser()
    
    # 4. [핵심 수정] LLM 스트리밍 실행 (astream 사용)
    try:
        # ainvoke 대신 astream을 사용합니다.
        # chunk는 LLM이 뱉어내는 한 글자(또는 토큰) 단위입니다.
        async for chunk in chain.astream({"context": context, "query": query}):
            yield chunk  # 실시간으로 조각을 던져줌
            
    except Exception as e:
        print(f"Generation Error: {e}")
        yield "답변 생성 중 오류가 발생했습니다."
        return

    # 5. 출처 표시 (답변이 다 끝난 뒤 마지막에 붙임)
    if used_docs:
        unique_sources = set()
        priority_docs = [d for d in used_docs if str(d.metadata.get('version', '')) in query]
        final_docs_to_show = priority_docs if priority_docs else used_docs

        for doc in final_docs_to_show:
            title = doc.metadata.get('report_title', '분석 보고서')
            date = doc.metadata.get('date', '')
            version = doc.metadata.get('version', '')
            
            source_parts = [f"- {title}"]
            if version: source_parts.append(f"**[v{version}]**")
            if date: source_parts.append(f"({date})")
            
            unique_sources.add(" ".join(source_parts))

        if unique_sources:
            sources_text = "\n".join(sorted(list(unique_sources)))
            # 마지막에 출처 정보를 yield
            yield f"\n\n---\n**📚 참고 문서**\n{sources_text}"