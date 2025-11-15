///
/// Copywrite(c) 2025 metanonia
///
/// AI-HUB 건강자료로 부터 encoder를 파인튜팅하기 위한 자료 생성
///

use anyhow::Result;
use glob::glob;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;

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

fn process_files(
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

                // 부정 질문 감지 및 정규화
                let (normalized_stem, is_negation) = normalize_question(&stem);

                // "1) xxx  2) xxx ..." 형태로 추출
                let mut choices = Vec::new();
                for cap in choice_re.split(rest) {
                    let trimmed = cap.trim();
                    if !trimmed.is_empty() {
                        choices.push(trimmed.to_string());
                    }
                }

                // 답 번호와 내용을 분리 (예: "4) 메타콜린..." → "메타콜린...")
                let answer_content = qa.answer.splitn(2, ')').nth(1).unwrap_or("").trim();

                let mut negative = Vec::new();
                let mut positive = String::new();

                if is_negation {
                    // 부정 질문: 정답이 "틀린 것/아닌 것"이므로
                    // 정답은 negative로, 나머지는 positive 후보로
                    for choice in &choices {
                        if choice.contains(answer_content) {
                            // 이것이 "틀린/아닌" 답이므로 실제로는 학습 시 회피해야 할 것
                            negative.push(choice.to_string());
                        } else {
                            // 나머지는 "맞는" 답들
                            if positive.is_empty() {
                                positive = choice.to_string();
                            } else {
                                // 추가 긍정 샘플도 저장 (첫 번째 외 나머지)
                                negative.push(choice.to_string());
                            }
                        }
                    }

                    // 부정 질문의 경우, 실제 오답(정답으로 표시된 것)을 negative의 맨 앞에 배치
                    // 이렇게 하면 학습 시 명시적으로 회피할 수 있음
                    if !negative.is_empty() {
                        negative.rotate_right(1);
                    }
                } else {
                    // 긍정 질문: 기존 로직
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
                // 주관식
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

    let mut output_samples_mc_train = Vec::new();
    let mut output_samples_nonmc_train = Vec::new();
    let mut output_samples_mc_val = Vec::new();
    let mut output_samples_nonmc_val = Vec::new();

    let choice_re = Regex::new(r"\d\)\s?")?;

    // Training 데이터 처리
    process_files(
        pattern_train,
        &mut output_samples_mc_train,
        &mut output_samples_nonmc_train,
        &choice_re,
    )?;

    // Validation 데이터 처리
    process_files(
        pattern_val,
        &mut output_samples_mc_val,
        &mut output_samples_nonmc_val,
        &choice_re,
    )?;

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
