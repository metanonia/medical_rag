import json
import os
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import ollama

DB_DIR = "./chroma_db"
COLLECTION_NAME = "medi_collection"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def query_chromadb(client, collection_name, query_text, model, top_k=5):
    """ChromaDB에서 유사도 검색"""
    collection = client.get_collection(name=collection_name)
    query_embedding = model.encode(query_text).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"type": "positive"}  # answer 타입만 검색
    )
    
    return results

def generate_answer_with_phi4(question, retrieved_contexts):
    """Phi4 모델로 답변 생성"""
    # 검색된 컨텍스트를 문자열로 조합
    context_text = "\n\n".join([f"[참고 {i+1}] {ctx}" for i, ctx in enumerate(retrieved_contexts)])
    
    # 프롬프트 구성
    prompt = f"""다음은 의료 지식 베이스에서 검색된 관련 정보입니다:

{context_text}

위 정보를 참고하여 다음 질문에 답변해주세요:

질문: {question}

답변:"""
    
    # Ollama를 통해 Phi4 모델 호출
    response = ollama.chat(
        model='phi4',
        messages=[
            {
                'role': 'system',
                'content': '당신은 의료 전문 AI 어시스턴트입니다. 제공된 참고 자료를 바탕으로 정확하고 전문적인 답변을 제공하세요.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    
    return response['message']['content']

def main():
    # 테스트 데이터 경로
    test_json_path = "models/embedding_val_nonmc.json"
    
    if not os.path.exists(test_json_path):
        print(f"Error: {test_json_path} not found.")
        return
    
    # 모델 로드
    model = SentenceTransformer("output/e5-base-medical-finetuned")
    
    # ChromaDB 클라이언트 생성

    client = PersistentClient(path=DB_DIR)
    
    # 테스트 데이터 로드
    test_samples = load_json(test_json_path)
    
    # 처음 5개 샘플만 테스트
    for idx, sample in enumerate(test_samples[:5], 1):
        question = sample["question"]
        correct_answer = sample["positive"]
        
        print(f"\n{'='*100}")
        print(f"[테스트 {idx}]")
        print(f"{'='*100}")
        
        # 1. 질문 출력
        print(f"\n📌 질문:\n{question}")
        
        # 2. 정답 출력
        print(f"\n✅ 정답:\n{correct_answer}")
        
        # 3. ChromaDB 검색
        print(f"\n🔍 ChromaDB 검색 결과 (Top 5):")
        search_results = query_chromadb(client, COLLECTION_NAME, question, model, top_k=5)
        
        retrieved_texts = []
        for i, (doc, distance) in enumerate(zip(search_results['documents'][0], search_results['distances'][0]), 1):
            print(f"\n  [{i}] 유사도: {1 - distance:.4f}")
            print(f"      {doc[:200]}..." if len(doc) > 200 else f"      {doc}")
            retrieved_texts.append(doc)
        
        # 4. Phi4로 답변 생성
        print(f"\n🤖 Phi4 생성 답변:")
        try:
            phi4_answer = generate_answer_with_phi4(question, retrieved_texts)
            print(phi4_answer)
        except Exception as e:
            print(f"Error: Phi4 호출 실패 - {e}")
        
        print(f"\n{'='*100}\n")

if __name__ == "__main__":
    main()
