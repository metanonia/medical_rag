## RAG 구축 

### [ 사용모델]
* 임베딩
  * multilingual-e5-base (https://huggingface.co/intfloat/multilingual-e5-base)

### [데이터]
AI HUB 공개 필수의료 의학지식 데이터
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

### [ Work Flow]
1. 데이터 전처리<br>
   (1) 질문형태에 따른 임베딩용 문서 생성<br>
        * AI HUB의 라벨링된 자료 사용
2. 임베딩<br>
   (1) 임베딩 모델 학습용 자료 생성<br>
       * cargo run --bin make_embedding_trading_data<br>
   (2) 임베딩 모델 학습<br>
        * python e5_tuning.py<br>
        * PYTORCH_ENABLE_MPS_FALLBACK=1 ACCELERATE_TORCH_DEVICE=cpu python e5_tuning.py<br>
   (3) 임베딩 정확도 측정<br>
        * python evaluate_embedding.py<br>
3. RAG

### 📜 라이선스
이 프로젝트는 AI Hub 데이터 이용 약관 및 각 모델 라이선스를 따릅니다.