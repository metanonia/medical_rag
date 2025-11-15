use anyhow::Result;
use glob::glob;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use csv::ReaderBuilder;

#[derive(Debug, Deserialize)]
struct QARecord {
    question: String,
    answer: String,
    q_type: u8,
}

#[derive(Debug, Serialize)]
struct OutputSampleMC {
    question: String,
    positive: String,
    negative: Vec<String>,
    is_negation: bool,
}

#[derive(Debug, Serialize)]
struct OutputSampleNonMC {
    question: String,
    positive: String,
}

fn remove_bom(content: &str) -> &str {
    content.strip_prefix('\u{FEFF}').unwrap_or(content)
}

fn normalize_question(question: &str) -> (String, bool) {
    let negation_patterns = [
        "아닌 것은",
        "아닌것은",
        "아니 것은",
        "틀린 것은",
        "틀린것은",
        "잘못된 것은",
        "잘못된것은",
        "제외한",
        "제외하면",
        "부적절한",
        "옳지 않은",
        "아니다",
        "거리가 먼",
        "관계없는",
    ];

    let is_negative = negation_patterns
        .iter()
        .any(|&pattern| question.contains(pattern));

    if is_negative {
        (format!("[부정질문] {}", question), true)
    } else {
        (question.to_string(), false)
    }
}

// === CSV 추가 처리 함수 ===
fn process_csv_file(
    path: &str,
    samples_mc: &mut Vec<OutputSampleMC>,
) -> Result<()> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    for result in rdr.records() {
        let record = result?;
        let question = record.get(4).unwrap_or("").trim();
        let a = record.get(5).unwrap_or("").trim();
        let b = record.get(6).unwrap_or("").trim();
        let c = record.get(7).unwrap_or("").trim();
        let d = record.get(8).unwrap_or("").trim();
        let e = record.get(9).unwrap_or("").trim();
        let choices = vec![a, b, c, d, e];
        let answer_raw = record.get(10).unwrap_or("").trim();

        // answer 변환 (A/B/C/D/E, 1~5 모두 지원)
        let ans_idx = match answer_raw.to_uppercase().as_str() {
            "A" => 0,
            "B" => 1,
            "C" => 2,
            "D" => 3,
            "E" => 4,
            v => v.parse::<usize>().unwrap_or(1) - 1,
        };
        let answer_content = choices.get(ans_idx).unwrap_or(&"").to_string();

        let (normalized_stem, is_negation) = normalize_question(&question);

        let mut negative = vec![];
        for (i, ch) in choices.iter().enumerate() {
            if i != ans_idx && !ch.is_empty() {
                negative.push(ch.to_string());
            }
        }
        let positive = answer_content;

        // 정답 존재 시만 추가
        if !positive.is_empty() {
            samples_mc.push(OutputSampleMC {
                question: normalized_stem,
                positive,
                negative,
                is_negation,
            });
        }
    }
    Ok(())
}

// process_files() (JSON)
fn process_json_files(
    pattern: &str,
    samples_mc: &mut Vec<OutputSampleMC>,
    samples_nonmc: &mut Vec<OutputSampleNonMC>,
    choice_re: &Regex,
) -> Result<()> {
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

                let (normalized_stem, is_negation) = normalize_question(&stem);

                let mut choices = Vec::new();
                for cap in choice_re.split(rest) {
                    let trimmed = cap.trim();
                    if !trimmed.is_empty() {
                        choices.push(trimmed.to_string());
                    }
                }

                let answer_content = qa.answer.splitn(2, ')').nth(1).unwrap_or("").trim();

                let mut negative = Vec::new();
                let mut positive = String::new();

                if is_negation {
                    for choice in &choices {
                        if choice.contains(answer_content) {
                            negative.push(choice.to_string());
                        } else {
                            if positive.is_empty() {
                                positive = choice.to_string();
                            } else {
                                negative.push(choice.to_string());
                            }
                        }
                    }
                    if !negative.is_empty() {
                        negative.rotate_right(1);
                    }
                } else {
                    for choice in &choices {
                        if choice.contains(answer_content) {
                            positive = choice.to_string();
                        } else {
                            negative.push(choice.to_string());
                        }
                    }
                }

                if !positive.is_empty() {
                    samples_mc.push(OutputSampleMC {
                        question: normalized_stem,
                        positive,
                        negative,
                        is_negation,
                    });
                }
            } else {
                samples_nonmc.push(OutputSampleNonMC {
                    question: qa.question.clone(),
                    positive: qa.answer.clone(),
                });
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let pattern_train = "data/Training/**/*.json";
    let pattern_val = "data/Validation/**/*.json";
    let csv_patterns_train = vec![
        "KorMedMCQA/data/*-train.csv", // 경로 상황에 맞게 조절
    ];
    let csv_patterns_val = vec![
        "KorMedMCQA/data/*-test.csv",  // 경로 상황에 맞게 조절
    ];

    let mut output_samples_mc_train = Vec::new();
    let mut output_samples_nonmc_train = Vec::new();
    let mut output_samples_mc_val = Vec::new();
    let mut output_samples_nonmc_val = Vec::new();

    let choice_re = Regex::new(r"\d\)\s?")?;

    // Training 데이터 처리 (JSON)
    process_json_files(
        pattern_train,
        &mut output_samples_mc_train,
        &mut output_samples_nonmc_train,
        &choice_re,
    )?;
    println!("step1 {}", output_samples_mc_train.len());
    // Validation 데이터 처리 (JSON)
    process_json_files(
        pattern_val,
        &mut output_samples_mc_val,
        &mut output_samples_nonmc_val,
        &choice_re,
    )?;
    println!("step2 {}", output_samples_mc_val.len());
    // Training 데이터 처리 (CSV)
    for csv_pattern in csv_patterns_train {
        for entry in glob(csv_pattern)? {
            if let Ok(path) = entry {
                process_csv_file(path.to_str().unwrap(), &mut output_samples_mc_train)?;
            }
        }
    }
    println!("step3 {}", output_samples_mc_train.len());
    // Validation 데이터 처리 (CSV)
    for csv_pattern in csv_patterns_val {
        for entry in glob(csv_pattern)? {
            if let Ok(path) = entry {
                process_csv_file(path.to_str().unwrap(), &mut output_samples_mc_val)?;
            }
        }
    }
    println!("step4 {}", output_samples_mc_val.len());

    // Training 파일 저장
    let export_mc_train = serde_json::to_string_pretty(&output_samples_mc_train)?;
    fs::write("models/embedding_train_mc.json", export_mc_train)?;

    let export_nonmc_train = serde_json::to_string_pretty(&output_samples_nonmc_train)?;
    fs::write("models/embedding_train_nonmc.json", export_nonmc_train)?;

    // Validation 파일 저장
    let export_mc_val = serde_json::to_string_pretty(&output_samples_mc_val)?;
    fs::write("models/embedding_val_mc.json", export_mc_val)?;

    let export_nonmc_val = serde_json::to_string_pretty(&output_samples_nonmc_val)?;
    fs::write("models/embedding_val_nonmc.json", export_nonmc_val)?;

    // 통계 출력
    let negation_count_train = output_samples_mc_train
        .iter()
        .filter(|s| s.is_negation)
        .count();
    let negation_count_val = output_samples_mc_val
        .iter()
        .filter(|s| s.is_negation)
        .count();

    println!("=== Training 데이터 ===");
    println!(
        "객관식: {}개 (긍정: {}개, 부정: {}개)",
        output_samples_mc_train.len(),
        output_samples_mc_train.len() - negation_count_train,
        negation_count_train
    );
    println!("주관식: {}개", output_samples_nonmc_train.len());

    println!("\n=== Validation 데이터 ===");
    println!(
        "객관식: {}개 (긍정: {}개, 부정: {}개)",
        output_samples_mc_val.len(),
        output_samples_mc_val.len() - negation_count_val,
        negation_count_val
    );
    println!("주관식: {}개", output_samples_nonmc_val.len());

    Ok(())
}
