///
/// Copywrite(c) 2025 metanonia
///
/// 허깅페이스 의료 시험 문제를 이용하여 평가용 자료 생성
///
use anyhow::Result;
use csv::ReaderBuilder;
use glob::glob;
use serde::Serialize;
use std::fs;

#[derive(Debug, Serialize)]
struct EvaluationSample {
    question: String,
    choices: Vec<String>,
    correct: String,
}

fn process_dev_csv(path: &str, samples: &mut Vec<EvaluationSample>) -> Result<()> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

    for result in rdr.records() {
        let record = result?;
        let question = record.get(4).unwrap_or("").trim();
        let a = record.get(5).unwrap_or("").trim();
        let b = record.get(6).unwrap_or("").trim();
        let c = record.get(7).unwrap_or("").trim();
        let d = record.get(8).unwrap_or("").trim();
        let e = record.get(9).unwrap_or("").trim();
        let answer_raw = record.get(10).unwrap_or("").trim();

        // 선택지를 "passage: ..." 형태로 변환
        let choices: Vec<String> = vec![a, b, c, d, e]
            .into_iter()
            .filter(|s| !s.is_empty())
            .map(|s| format!("passage: {}", s))
            .collect();

        // 정답 문자열 자체를 찾기
        // 우선 answer_raw가 A~E 또는 1~5라면 인덱스로 변환 후 choices에서 답 선택
        // 아니면 answer_raw 자체를 정답으로 사용
        let correct = match answer_raw.to_uppercase().as_str() {
            "A" => choices.get(0).cloned().unwrap_or_default(),
            "B" => choices.get(1).cloned().unwrap_or_default(),
            "C" => choices.get(2).cloned().unwrap_or_default(),
            "D" => choices.get(3).cloned().unwrap_or_default(),
            "E" => choices.get(4).cloned().unwrap_or_default(),
            v => {
                let idx = v.parse::<usize>().unwrap_or(0);
                if idx >= 1 && idx <= choices.len() {
                    choices.get(idx - 1).cloned().unwrap_or_default()
                } else {
                    format!("passage: {}", answer_raw)
                }
            }
        };

        // question도 "query: " 접두사 추가 (옵션)
        let formatted_question = format!("query: {}", question);

        samples.push(EvaluationSample {
            question: formatted_question,
            choices,
            correct,
        });
    }

    Ok(())
}


fn main() -> Result<()> {
    let csv_pattern_dev = "KorMedMCQA/data/*-dev.csv";
    let mut evaluation_samples = Vec::new();

    // dev.csv 파일들 처리
    for entry in glob(csv_pattern_dev)? {
        if let Ok(path) = entry {
            println!("Processing: {:?}", path);
            process_dev_csv(path.to_str().unwrap(), &mut evaluation_samples)?;
        }
    }

    // JSON 파일로 저장
    let export_eval = serde_json::to_string_pretty(&evaluation_samples)?;
    fs::write("models/embedding_evaluation.json", export_eval)?;

    println!("\n=== 평가 데이터 ===");
    println!("총 {}개 문항", evaluation_samples.len());

    Ok(())
}
