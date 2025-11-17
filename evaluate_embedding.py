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

def evaluate_model_detailed(model, test_cases, verbose=True):
    correct = 0
    total = len(test_cases)
    score_gaps = []
    
    # 부정 질문 패턴
    negation_patterns = [
        "아닌 것", "아닌것", "아니 것", "틀린 것", "틀린것",
        "잘못된 것", "잘못된것", "제외한", "제외하면",
        "부적절한", "옳지 않은", "아니다", "거리가 먼", "관계없는",
        "해당하지 않는", "해당되지 않는", "적절하지 않은"
    ]

    for idx, case in enumerate(test_cases):
        if verbose:
            print(f"\n{'='*80}")
            print(f"문제 {idx + 1}/{total}")
            print(f"{'='*80}")
            
            # 질문 출력
            print(f"\n질문: {case['question']}")
            
        # 부정 질문 판단
        is_negation = any(pattern in case['question'] for pattern in negation_patterns)
        
        if verbose and is_negation:
            print("⚠️ 부정 질문 감지 - 가장 낮은 유사도를 정답으로 선택")
            
        # 임베딩 및 유사도 계산
        q_emb = model.encode(case["question"]).reshape(1, -1)
        c_embs = model.encode(case["choices"])
        similarities = cosine_similarity(q_emb, c_embs)[0]

        # correct가 텍스트인 경우 인덱스로 변환
        if isinstance(case['correct'], str):
            try:
                correct_idx = case['choices'].index(case['correct'])
            except ValueError:
                print(f"경고: 정답을 찾을 수 없음 - {case['correct']}")
                continue
        else:
            correct_idx = case['correct']
        
        # 예측: 부정 질문이면 최소값, 긍정 질문이면 최대값
        if is_negation:
            predicted = np.argmin(similarities)
            sorted_indices = np.argsort(similarities)  # 오름차순
        else:
            predicted = np.argmax(similarities)
            sorted_indices = np.argsort(similarities)[::-1]  # 내림차순
        
        if verbose:
            print(f"\n선택지 ({'낮은' if is_negation else '높은'} 유사도 순):")
            for rank, i in enumerate(sorted_indices, 1):
                choice = case["choices"][i]
                sim = similarities[i]
                is_correct_answer = (i == correct_idx)
                marker = "✓ 정답" if is_correct_answer else ""
                print(f"  {rank}위. [{i+1}번] (유사도: {sim:.4f}) {choice} {marker}")

        # 예측 결과
        is_correct = predicted == correct_idx
        
        if verbose:
            print(f"\n예측: 선택지 {predicted + 1}번")
            print(f"정답: 선택지 {correct_idx + 1}번")
            print(f"결과: {'✓ 정답' if is_correct else '✗ 오답'}")
        
        if is_correct:
            correct += 1

        # 점수 차이 계산
        sorted_scores = np.sort(similarities)[::-1]
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        score_gaps.append(gap)
        
        if verbose:
            if is_negation:
                print(f"최저-차순위 점수 차이: {gap:.4f}")
            else:
                print(f"1위-2위 점수 차이: {gap:.4f}")

    # 최종 통계
    print(f"\n{'='*80}")
    print(f"최종 결과")
    print(f"{'='*80}")
    print(f"정확도: {correct}/{total} ({100*correct/total:.2f}%)")
    if score_gaps:
        print(f"평균 점수 차이: {np.mean(score_gaps):.4f}")
        print(f"점수 차이 표준편차: {np.std(score_gaps):.4f}")
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'score_gaps': score_gaps,
        'mean_gap': np.mean(score_gaps) if score_gaps else 0,
        'std_gap': np.std(score_gaps) if score_gaps else 0
    }


# 평가 실행
evaluate_model_detailed(model, test_cases, verbose=True)
