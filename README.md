## RAG 구축 프로세스 학습

### [ 사용모델]
* 임베딩
  * multilingual-e5-base (https://huggingface.co/intfloat/multilingual-e5-base)

### [데이터]
#### 과학기술정보통신부와 한국지능정보사회진흥원이 운영하는 AI HUB의 공개 필수의료 의학지식 데이터
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71875
1. 데이터 구성<br>
   (1) qa_id: 질의응답별 고유 id<br>
   (2) domain: 의료분야<br>
   (3) q_type: 질문 유형<br>
   (4) question: 질문<br>
   (5) answer: 답변<br>
2. 어노테이션 포맷<br>
   (1) qa_id: 1-보라매병원, 2-삼성서울병원, 3-서울대병원, 4-서울성모병원, 5-세브란스병원, 6-크라우드웍스<br>
   (2) domain: 1-외과, 2-예방의학, 3-정신건강의학과, 4-신경과/신경외과, 5-피부과, 6-안과, 7-이비인후과, 8-비뇨의학과, 9-방사선종양학과, 10-병리과, 11-마취통증의학과, 12-의료법규, 13-기타, 14-산부인과, 15-소아청소년과, 16-응급의학과, 17-내과<br>
   (3) q_type: 1-객관식, 2-단답형, 3-서술형<br>
   (4) question: 질문 텍스트<br>
   (5) answer: 답변 텍스트<br>

##### 허깅페이스에 공개된 의학 시험 정보
git clone https://huggingface.co/datasets/sean0042/KorMedMCQA<br>
* subject: doctor, nurse, or pharm
* year: year of the examination
* period: period of the examination
* q_number: question number of the examination
* question: question
* A: First answer choice
* B: Second answer choice
* C: Third answer choice
* D: Fourth answer choice
* E: Fifth answer choice
* cot : Answer with reasoning annotated by professionals (only available in fewshot split)
* answer : Answer (1 to 5). 1 denotes answer A, and 5 denotes answer E

### [ Work Flow]
#### 1. 데이터 전처리
  - 라벨링 작업: 질문형태의 임베딩용 문서 생성<br>
    - AI HUB 및 Hugingface의 라벨링된 자료 사용<br>
    - RAG에서 가장 중요한 것은 1) 청크 나누기 2) 임베딩 모델의 파인 튜닝<br>
    - 질문형 문서 개별 생성 작업이 힘든 경우, 청크 분할 후, 다음 청크를 포지티브 답변으로 가정하여 질문 문서 생성<br>
#### 2. 임베딩 모델 학습<br>
(1) 임베딩 모델 학습용 자료 생성<br>
  ```python make_embedding_trading_data.py```<br>

(2) 임베딩 모델 학습<br>
  ```python e5_tuning.py```<br>
  참고: ```PYTORCH_ENABLE_MPS_FALLBACK=1 ACCELERATE_TORCH_DEVICE=cpu python e5_tuning.py```<br>

(3) 임베딩 정확도 측정<br>
  ```python make_evaluattion_data.py```<br>
  ```python evaluate_embedding.py```<br>
  - 객관식 문제 풀이에서는 질문과 선택지를 각각 임베딩한다음, 질문과 각 항목간의 유사도 검색하여 정답 검출<br>
  - KorMedMCQA/data/*-dev.csv 사용 테스트<br>

#### 3. RAG<br>
(1) 벡터디비 생성<br>
  - ChromaDB 사용<br>
  ```python make_vectordb.py```<br>

(2) ollama(phi4) 연동<br>
  - 주관식: 질문을 임베딩한 다음, 코사인 유사도를 기준으로 answer(positive) 검색, 질문과 검색내용을 LLM으로 전달<br>
  - 학습에 사용한 embedding_val_nonmc.json 이용 테스트<br>
  ```python rag_test.py```
