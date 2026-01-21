import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 설정값 (기존 코드와 동일하게)
PERSIST_DIRECTORY = "./RAG/chroma_advanced_db"
embeddings = OllamaEmbeddings(model="bge-m3")

# DB 로드
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# 1. 전체 데이터 개수 확인
print(f"📊 전체 문서 개수: {vector_store._collection.count()}")

# 2. '쿠팡플레이'라는 이름의 메타데이터를 가진 데이터 검색
# where 조건으로 메타데이터 필터링 확인
results = vector_store.get(where={"app_name": "쿠팡플레이"})

if results["documents"]:
    print(f"✅ '쿠팡플레이' 데이터가 {len(results['documents'])}개 존재합니다.")
    print(f"📄 첫 번째 데이터 샘플: {results['documents'][0][:100]}...")
    print(f"🏷️ 메타데이터 샘플: {results['metadatas'][0]}")
else:
    print("❌ '쿠팡플레이' 관련 데이터를 찾을 수 없습니다. 메타데이터 이름을 확인하세요.")

# 3. 모든 앱 이름 목록 확인 (중복 제거)
all_metas = vector_store.get()["metadatas"]
app_names = set(m.get("app_name") for m in all_metas if m.get("app_name"))
print(f"📱 DB에 등록된 모든 앱 목록: {app_names}")