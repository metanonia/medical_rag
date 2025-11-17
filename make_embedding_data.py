import glob
import json
import re
import os

# 정규 표현식 패턴
choice_re = re.compile(r"\d\)\s?")

# 부정 질문 패턴 목록
negation_patterns = [
    "아닌 것은", "아닌것은", "아니 것은", "틀린 것은", "틀린것은",
    "잘못된 것은", "잘못된것은", "제외한", "제외하면",
    "부적절한", "옳지 않은", "아니다", "거리가 먼", "관계없는"
]

def remove_bom(content):
    """UTF-8 BOM 제거"""
    return content.lstrip('\ufeff')

def normalize_question(question):
    """질문 정규화 및 부정 질문 판단"""
    is_negation = any(pattern in question for pattern in negation_patterns)
    if is_negation:
        return f"[부정질문] {question}", True
    else:
        return question, False

def process_csv_file(path, samples_mc):
    """CSV 파일 처리 (인덱스 기반)"""
    import csv
    with open(path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 헤더 건너뛰기

        for row in reader:
            if len(row) < 11:
                continue

            question = row[4].strip()
            a = row[5].strip()
            b = row[6].strip()
            c = row[7].strip()
            d = row[8].strip()
            e = row[9].strip()
            choices = [a, b, c, d, e]
            answer_raw = row[10].strip()

            # answer 변환 (A/B/C/D/E, 1~5 모두 지원)
            answer_upper = answer_raw.upper()
            if answer_upper == "A":
                ans_idx = 0
            elif answer_upper == "B":
                ans_idx = 1
            elif answer_upper == "C":
                ans_idx = 2
            elif answer_upper == "D":
                ans_idx = 3
            elif answer_upper == "E":
                ans_idx = 4
            else:
                try:
                    ans_idx = int(answer_raw) - 1
                except:
                    ans_idx = 0

            if ans_idx < 0 or ans_idx >= len(choices):
                continue

            answer_content = choices[ans_idx]

            normalized_stem, is_negation = normalize_question(question)

            negative = []
            for i, ch in enumerate(choices):
                if i != ans_idx and ch:
                    negative.append(ch)

            positive = answer_content

            # 정답 존재 시만 추가
            if positive:
                samples_mc.append({
                    "question": normalized_stem,
                    "positive": positive,
                    "negative": negative,
                    "is_negation": is_negation
                })

def process_json_files(pattern, samples_mc, samples_nonmc):
    """JSON 파일 처리"""
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                content = remove_bom(content)
                qa = json.loads(content)

                question = qa.get('question', '')
                answer = qa.get('answer', '')
                q_type = qa.get('q_type', 0)

                if q_type == 1:
                    # 객관식 처리
                    parts = question.split('?', 1)
                    if len(parts) < 2:
                        continue

                    stem = parts[0].strip() + '?'
                    rest = parts[1]

                    normalized_stem, is_negation = normalize_question(stem)

                    # 선택지 추출
                    choices = []
                    for cap in choice_re.split(rest):
                        trimmed = cap.strip()
                        if trimmed:
                            choices.append(trimmed)

                    # 정답 내용 추출
                    if ')' in answer:
                        answer_content = answer.split(')', 1)[1].strip()
                    else:
                        answer_content = answer.strip()

                    negative = []
                    positive = ""

                    if is_negation:
                        # 부정 질문: 정답이 negative, 나머지 중 하나가 positive
                        for choice in choices:
                            if answer_content in choice:
                                negative.append(choice)
                            else:
                                if not positive:
                                    positive = choice
                                else:
                                    negative.append(choice)

                        # negative 회전 (Rust의 rotate_right(1) 구현)
                        if negative:
                            negative = [negative[-1]] + negative[:-1]
                    else:
                        # 긍정 질문: 정답이 positive, 나머지가 negative
                        for choice in choices:
                            if answer_content in choice:
                                positive = choice
                            else:
                                negative.append(choice)

                    if positive:
                        samples_mc.append({
                            "question": normalized_stem,
                            "positive": positive,
                            "negative": negative,
                            "is_negation": is_negation
                        })
                else:
                    # 주관식
                    samples_nonmc.append({
                        "question": question,
                        "positive": answer
                    })
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

def main():
    pattern_train_json = "data/Training/**/*.json"
    pattern_val_json = "data/Validation/**/*.json"
    csv_patterns_train = [
        "KorMedMCQA/data/*-train.csv",
        "KorMedMCQA/data/*-dev.csv"
    ]
    csv_patterns_val = ["KorMedMCQA/data/*-test.csv"]

    samples_mc_train = []
    samples_nonmc_train = []
    samples_mc_val = []
    samples_nonmc_val = []

    # Training JSON 처리
    process_json_files(pattern_train_json, samples_mc_train, samples_nonmc_train)
    print(f"step1 {len(samples_mc_train)}")

    # Validation JSON 처리
    process_json_files(pattern_val_json, samples_mc_val, samples_nonmc_val)
    print(f"step2 {len(samples_mc_val)}")

    # Training CSV 처리
    for csv_pattern in csv_patterns_train:
        for filepath in glob.glob(csv_pattern):
            process_csv_file(filepath, samples_mc_train)
    print(f"step3 {len(samples_mc_train)}")

    # Validation CSV 처리
    for csv_pattern in csv_patterns_val:
        for filepath in glob.glob(csv_pattern):
            process_csv_file(filepath, samples_mc_val)
    print(f"step4 {len(samples_mc_val)}")

    # 결과 저장
    os.makedirs("models", exist_ok=True)

    with open("models/embedding_train_mc.json", 'w', encoding='utf-8') as f:
        json.dump(samples_mc_train, f, ensure_ascii=False, indent=2)

    with open("models/embedding_train_nonmc.json", 'w', encoding='utf-8') as f:
        json.dump(samples_nonmc_train, f, ensure_ascii=False, indent=2)

    with open("models/embedding_val_mc.json", 'w', encoding='utf-8') as f:
        json.dump(samples_mc_val, f, ensure_ascii=False, indent=2)

    with open("models/embedding_val_nonmc.json", 'w', encoding='utf-8') as f:
        json.dump(samples_nonmc_val, f, ensure_ascii=False, indent=2)

    # 통계 출력
    negation_train = sum(1 for s in samples_mc_train if s['is_negation'])
    negation_val = sum(1 for s in samples_mc_val if s['is_negation'])

    print("\n=== Training 데이터 ===")
    print(f"객관식: {len(samples_mc_train)}개 (긍정: {len(samples_mc_train)-negation_train}개, 부정: {negation_train}개)")
    print(f"주관식: {len(samples_nonmc_train)}개")

    print("\n=== Validation 데이터 ===")
    print(f"객관식: {len(samples_mc_val)}개 (긍정: {len(samples_mc_val)-negation_val}개, 부정: {negation_val}개)")
    print(f"주관식: {len(samples_nonmc_val)}개")

if __name__ == "__main__":
    main()
