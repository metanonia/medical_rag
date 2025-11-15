import os
import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['ACCELERATE_TORCH_DEVICE'] = 'cpu'

if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

# ============================================
# 1. 학습 데이터 로드
# ============================================
# 객관식 데이터 (negative 포함)
with open("./embedding_train_mc.json", "r", encoding="utf-8") as f:
    train_data_mc = json.load(f)

# 주관식 데이터 (negative 없음)
with open("./embedding_train_nonmc.json", "r", encoding="utf-8") as f:
    train_data_nonmc = json.load(f)

print(f"학습 - 객관식 데이터: {len(train_data_mc)}")
print(f"학습 - 주관식 데이터: {len(train_data_nonmc)}")
print(f"학습 - 전체 데이터: {len(train_data_mc) + len(train_data_nonmc)}")

# ============================================
# 2. 검증 데이터 로드
# ============================================
# 객관식 검증 데이터
with open("./embedding_val_mc.json", "r", encoding="utf-8") as f:
    val_data_mc = json.load(f)

# 주관식 검증 데이터
with open("./embedding_val_nonmc.json", "r", encoding="utf-8") as f:
    val_data_nonmc = json.load(f)

print(f"\n검증 - 객관식 데이터: {len(val_data_mc)}")
print(f"검증 - 주관식 데이터: {len(val_data_nonmc)}")
print(f"검증 - 전체 데이터: {len(val_data_mc) + len(val_data_nonmc)}")

# ============================================
# 3. 학습 데이터 준비 (InputExample 변환)
# ============================================
train_examples = []

# 객관식 데이터 처리 (negative 있음)
for item in train_data_mc:
    question = f"query: {item['question']}"
    positive = f"passage: {item['positive']}"

    # Positive 쌍 추가
    train_examples.append(InputExample(texts=[question, positive]))

    # Negative 샘플들도 추가 (선택적이지만 효과적)
    # MultipleNegativesRankingLoss는 배치 내에서도 negative를 만들지만
    # 명시적 negative가 있으면 더 좋음
    negatives = item.get("negative", [])
    for neg in negatives[:2]:  # 너무 많으면 메모리 부담, 상위 2개만
        train_examples.append(InputExample(texts=[question, f"passage: {neg}"]))

# 주관식 데이터 처리 (negative 없음)
for item in train_data_nonmc:
    question = f"query: {item['question']}"
    positive = f"passage: {item['positive']}"

    train_examples.append(InputExample(texts=[question, positive]))

print(f"\n총 학습 샘플 수: {len(train_examples)}")

# ============================================
# 4. 모델 로드
# ============================================
device = torch.device("cpu")
model = SentenceTransformer('intfloat/multilingual-e5-base').to(device)
# 로컬 모델: model = SentenceTransformer('./local_models/e5-base')

print("\nCUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("Current device:", device)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ============================================
# 5. DataLoader 구성
# ============================================
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16
)

# ============================================
# 6. 손실 함수 설정
# ============================================
train_loss = losses.MultipleNegativesRankingLoss(model)

# ============================================
# 7. 검증 데이터 준비
# ============================================
# 객관식 + 주관식 검증 데이터 통합
val_data_all = val_data_mc + val_data_nonmc
val_sentences1 = []
val_sentences2 = []
val_scores = []

# 1. 정답 쌍(유사도 1.0)
for item in val_data_all:
    val_sentences1.append(f"query: {item['question']}")
    val_sentences2.append(f"passage: {item['positive']}")
    val_scores.append(1.0)

# 2. 오답 쌍(유사도 0.0, 객관식만 해당)
for item in val_data_mc:
    negatives = item.get("negative", [])
    for neg in negatives[:2]:  # 너무 많을 필요 없음, 2개 추천
        val_sentences1.append(f"query: {item['question']}")
        val_sentences2.append(f"passage: {neg}")
        val_scores.append(0.0)

print(f"검증용 쌍: {len(val_sentences1)}개 (정답 {len(val_data_all)}, 오답 {len(val_sentences1)-len(val_data_all)})")

evaluator = EmbeddingSimilarityEvaluator(
    val_sentences1,
    val_sentences2,
    val_scores,
    name="validation"
)

# ============================================
# 8. 파인튜닝 실행
# ============================================
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=500,
    warmup_steps=100,
    output_path="./output/e5-base-medical-finetuned",
    save_best_model=True,
    show_progress_bar=True,
)

print("\n파인튜닝 완료!")
print("모델 저장 위치: ./output/e5-base-medical-finetuned")

# ============================================
# 9. 파인튜닝된 모델 테스트
# ============================================
finetuned_model = SentenceTransformer("./output/e5-base-medical-finetuned")

# 테스트 케이스
test_cases = [
    ("query: 70세 여성이 반복적인 실신을 호소하며 병원에 내원했습니다. 심전도 검사에서 좌각차단(bifascicular block) 소견이 확인되었으며, 추가 검사에서 간헐적인 3도 방실차단이 관찰되었습니다. 이 환자에게 가장 적절한 치료는 무엇입니까", "passage: 영구형 인공심박조율기 삽입"),
    ("query: 자궁내막증 진단 방법은?", "passage: 복강경 검사"),
]

from sentence_transformers.util import cos_sim

print("\n=== 테스트 결과 ===")
for query, doc in test_cases:
    query_emb = finetuned_model.encode(query)
    doc_emb = finetuned_model.encode(doc)
    similarity = cos_sim(query_emb, doc_emb)
    print(f"Q: {query[:30]}...")
    print(f"D: {doc[:30]}...")
    print(f"유사도: {similarity.item():.4f}\n")
