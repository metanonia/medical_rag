from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# 테스트 케이스 리스트
# embedding_evaluation.json
# [
#  {
#    "question": "...",
#    "choices": [
#        "passage: ...",
#        "passage: ...",
#        ...
#    ],
#    "correct": ..
#  },
#]
with open("models/embedding_evaluation.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

# 임베딩 모델 로드
model_path = "models/output/e5-base-medical-finetuned"
model = SentenceTransformer(model_path)

def evaluate_model(model, test_cases):
    correct = 0
    total = len(test_cases)
    score_gaps = []

    for case in test_cases:
        q_emb = model.encode(case["question"]).reshape(1, -1)
        c_embs = model.encode(case["choices"])
        similarities = cosine_similarity(q_emb, c_embs)[0]

        predicted = np.argmax(similarities)
        if predicted == case["correct"]:
            correct += 1

        sorted_scores = np.sort(similarities)[::-1]
        score_gaps.append(sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0)

    accuracy = correct / total
    avg_gap = np.mean(score_gaps)

    print(f"정확도: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"평균 유사도 차이: {avg_gap:.4f}")

    if accuracy >= 0.90 and avg_gap >= 0.1:
        print("파인튜닝 불필요: 임베딩 품질 충분함")
    elif accuracy >= 0.85 and avg_gap >= 0.05:
        print("파인튜닝 선택적: 현재도 사용 가능하나 개선 여지 있음")
    else:
        print("파인튜닝 필요: 품질 개선 권장")

# 평가 실행
evaluate_model(model, test_cases)
