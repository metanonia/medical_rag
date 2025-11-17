from huggingface_hub import snapshot_download

# 모든 파일 다운로드 (인터넷 연결 없이 사용 가능)
snapshot_download(
    repo_id="intfloat/multilingual-e5-base",
    local_dir="models/e5-base-offline",
    local_dir_use_symlinks=False  # 심볼릭 링크 없이 실제 파일 복사
)