///
/// Copywrite(c) 2025 metanonia
///
/// AI-HUB 거강자료로 부터 encoder를 파인튜팅하기 위한 자료 생성
/// Trading Data만 처리
///

use serde::{Deserialize, Serialize};
use std::fs;
use glob::glob;
use regex::Regex;

#[derive(Debug, Deserialize)]
struct QARecord {
    qa_id: u64,
    domain: u64,
    q_type: u8,
    question: String,
    answer: String,
}

#[derive(Serialize)]
struct OutputSampleMC {
    question: String,
    positive: String,
    negative: Vec<String>,
}

#[derive(Serialize)]
struct OutputSampleNonMC {
    question: String,
    positive: String,
}

fn remove_bom(s: &str) -> &str {
    if s.starts_with("\u{feff}") {
        &s[3..]
    } else {
        s
    }
}

fn main() -> anyhow::Result<()> {
    let pattern = "data/Training/**/*.json";
    let mut output_samples_mc = Vec::new();
    let mut output_samples_nonmc = Vec::new();
    let choice_re = Regex::new(r"\d\)\s?")?;

    for entry in glob(pattern)? {
        if let Ok(path) = entry {
            let content = fs::read_to_string(&path)?;
            let content = remove_bom(&content);
            let qa: QARecord = serde_json::from_str(&content)?;

            if qa.q_type == 1 {
                // 객관식 처리: 선택지 추출
                let q_parts: Vec<&str> = qa.question.splitn(2, '?').collect();
                let stem = q_parts.get(0).unwrap_or(&"").trim().to_string() + "?";
                let rest = q_parts.get(1).unwrap_or(&"");

                // "1) xxx  2) xxx ..." 형태로 추출
                let mut choices = Vec::new();
                for cap in choice_re.split(rest) {
                    let trimmed = cap.trim();
                    if !trimmed.is_empty() {
                        choices.push(trimmed.to_string());
                    }
                }
                // 답 번호와 내용을 분리 (예: "4) 메타콜린..." → "메타콜린...")
                let answer_content = qa.answer
                    .splitn(2, ')')
                    .nth(1)
                    .unwrap_or("")
                    .trim();

                let mut negative = Vec::new();
                let mut positive = String::new();
                for choice in &choices {
                    // choice가 "메타콜린 기관지유발검사" 등 정답 내용과 매칭되는지
                    // 번호 제외하고 비교 (둘 다 trim)
                    if choice.contains(answer_content) {
                        positive = choice.to_string();
                    } else {
                        negative.push(choice.to_string());
                    }
                }

                if !positive.is_empty() {
                    output_samples_mc.push(OutputSampleMC {
                        question: stem,
                        positive,
                        negative,
                    });
                }
            } else {
                // 주관식
                output_samples_nonmc.push(OutputSampleNonMC {
                    question: qa.question.clone(),
                    positive: qa.answer.clone(),
                });
            }
        }
    }

    // 파일 저장
    let export_mc = serde_json::to_string_pretty(&output_samples_mc)?;
    fs::write("models/embedding_train_mc.json", export_mc)?;

    let export_nonmc = serde_json::to_string_pretty(&output_samples_nonmc)?;
    fs::write("models/embedding_train_nonmc.json", export_nonmc)?;

    println!(
        "객관식: {}개, 주관식: {}개",
        output_samples_mc.len(),
        output_samples_nonmc.len()
    );
    Ok(())
}
