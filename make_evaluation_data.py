import glob
import csv
import json
import os

def process_dev_csv(path, samples):
    """dev.csv 파일 처리 (인덱스 기반)"""
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
            answer_raw = row[10].strip()

            # 선택지를 "passage: ..." 형태로 변환
            choices = []
            for ch in [a, b, c, d, e]:
                if ch:
                    choices.append(f"passage: {ch}")

            # 정답 결정
            correct = ""
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
                # 숫자로 파싱 시도
                try:
                    idx = int(answer_raw)
                    if 1 <= idx <= len(choices):
                        ans_idx = idx - 1
                    else:
                        ans_idx = -1
                except:
                    ans_idx = -1

            # 정답 선택
            if ans_idx >= 0 and ans_idx < len(choices):
                correct = choices[ans_idx]
            else:
                correct = f"passage: {answer_raw}"

            # question에 "query: " 접두사 추가
            formatted_question = f"query: {question}"

            samples.append({
                "question": formatted_question,
                "choices": choices,
                "correct": correct
            })

def main():
    csv_pattern_dev = "KorMedMCQA/data/*-dev.csv"
    evaluation_samples = []

    # dev.csv 파일들 처리
    for filepath in glob.glob(csv_pattern_dev):
        print(f"Processing: {filepath}")
        process_dev_csv(filepath, evaluation_samples)

    # 결과 저장
    os.makedirs("models", exist_ok=True)

    with open("models/embedding_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation_samples, f, ensure_ascii=False, indent=2)

    print("\n=== 평가 데이터 ===")
    print(f"총 {len(evaluation_samples)}개 문항")

if __name__ == "__main__":
    main()
