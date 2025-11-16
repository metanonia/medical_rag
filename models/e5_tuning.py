import os
import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.util import cos_sim
from tqdm import tqdm

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

    # Negative 샘플들도 추가
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

class IREvaluator(SentenceEvaluator):
    """
    IR 평가를 위한 커스텀 Evaluator
    - Query → 여러 Passage
    - 각 Query에 대해 correct_passage를 지정
    - MRR / Recall@k 계산
    """

    def __init__(self, queries, passages, relevant_ids, name='ir-eval'):
        """
        queries:     List[str]             (평가할 query 리스트)
        passages:    List[str]             (전체 후보 passage 리스트, 고정)
        relevant_ids: List[int]            (각 query별 정답 passage의 index)
        """
        self.queries = queries
        self.passages = passages
        self.relevant_ids = relevant_ids
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):

        print(f"\n==== IR Evaluation (epoch={epoch}, step={steps}) ====")

        # Passage 임베딩 미리 계산
        print("Passage embedding...")
        passage_emb = model.encode(self.passages, convert_to_tensor=True, show_progress_bar=True)

        mrr_total = 0.0
        recall1 = 0
        recall5 = 0
        recall10 = 0

        print("Query scoring...")
        for idx, query in enumerate(tqdm(self.queries)):
            q_emb = model.encode(query, convert_to_tensor=True)

            # 모든 패시지와 코사인 유사도 계산
            sims = cos_sim(q_emb, passage_emb)[0]  # shape: (num_passage,)

            # 내림차순 랭킹
            rank = torch.argsort(sims, descending=True)

            # 정답 passage index
            relevant = self.relevant_ids[idx]

            # rank에서 정답 위치 찾기
            rank_position = (rank == relevant).nonzero(as_tuple=True)[0].item()
            rank_position_1based = rank_position + 1

            # MRR
            mrr_total += 1.0 / rank_position_1based

            # Recall@k
            if rank_position < 1:
                recall1 += 1
            if rank_position < 5:
                recall5 += 1
            if rank_position < 10:
                recall10 += 1

        num_queries = len(self.queries)
        mrr = mrr_total / num_queries
        r1 = recall1 / num_queries
        r5 = recall5 / num_queries
        r10 = recall10 / num_queries

        print(f"MRR: {mrr:.4f}")
        print(f"Recall@1:  {r1:.4f}")
        print(f"Recall@5:  {r5:.4f}")
        print(f"Recall@10: {r10:.4f}")
        print("============================================\n")

        # SentenceTransformers는 float 리턴 → save_best_model 기준점
        return mrr

# ============================================
# 5. DataLoader 구성
# ============================================
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=32
)

# ============================================
# 6. 손실 함수 설정
# ============================================
train_loss = losses.MultipleNegativesRankingLoss(model)

# ============================================
# 7. 검증 데이터 준비 (정답 및 오답을 함께 검증 평가해야 함)
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

# STS용 (의미론적 유사도 측정.. 이 셈플에선 이런식으로 평가를 하면 안됨)
evaluator = EmbeddingSimilarityEvaluator(
    val_sentences1,
    val_sentences2,
    val_scores,
    name="validation"
)

# === IR 평가 데이터 구성 ===

# 전체 passage 리스트
passages = []
passage_map = {}  # 중복 방지 (text → index)

# Query 리스트
queries = []
relevant_ids = []

# 1) Positive/Negative 포함한 통합 passage 리스트 생성
def get_pid(p):
    if p not in passage_map:
        passage_map[p] = len(passages)
        passages.append(p)
    return passage_map[p]

# 2) Query 처리
for item in val_data_mc + val_data_nonmc:
    q = f"query: {item['question']}"
    queries.append(q)

    pos_text = f"passage: {item['positive']}"
    pos_id = get_pid(pos_text)

    # 정답 passage index
    relevant_ids.append(pos_id)

    # Negative 도 passage pool에 추가
    for neg in item.get("negative", []):
        get_pid(f"passage: {neg}")

print(f"Total passages: {len(passages)}")
print(f"Total queries: {len(queries)}")

# === IR Evaluator 연결 ===
ir_evaluator = IREvaluator(
    queries=queries,
    passages=passages,
    relevant_ids=relevant_ids,
)

# ============================================
# 8. 파인튜닝 실행
# ============================================
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator,
    epochs=3,
    evaluation_steps=500,
    warmup_steps=100,
    output_path="./output/e5-base-medical-finetuned",
    save_best_model=True,              # validation 최고 모델 저장 유지
    show_progress_bar=True,
    optimizer_params={
        'lr': 1e-5,
        'weight_decay': 0.01
    },
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

print("\n=== 테스트 결과 ===")
for query, doc in test_cases:
    query_emb = finetuned_model.encode(query)
    doc_emb = finetuned_model.encode(doc)
    similarity = cos_sim(query_emb, doc_emb)
    print(f"Q: {query[:30]}...")
    print(f"D: {doc[:30]}...")
    print(f"유사도: {similarity.item():.4f}\n")
