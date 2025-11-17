import json
import os
import shutil
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

DB_DIR = "./chroma_db"
COLLECTION_NAME = "medi_collection"
BATCH_SIZE = 5000  # 안전한 배치 크기

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def reset_chroma():
    try:
        shutil.rmtree(DB_DIR)
        print("Old Chroma DB removed.")
    except FileNotFoundError:
        print("No previous Chroma DB found. Starting fresh.")

def count_total_samples(paths_mc, paths_nonmc):
    total = 0
    for path in paths_mc + paths_nonmc:
        if os.path.exists(path):
            total += len(load_json(path))
    return total

def flush_batch(collection, documents, metadatas, ids, embeddings):
    """배치로 ChromaDB에 저장"""
    if not ids:
        return

    print(f"Flushing batch to ChromaDB... (size={len(ids)})")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    documents.clear()
    metadatas.clear()
    ids.clear()
    embeddings.clear()

def main():
    json_paths_mc = [
        "models/embedding_train_mc.json",
        "models/embedding_val_mc.json",
    ]
    json_paths_nonmc = [
        "models/embedding_train_nonmc.json",
        "models/embedding_val_nonmc.json",
    ]

    model = SentenceTransformer("models/output/e5-base-medical-finetuned")

    reset_chroma()

    client = PersistentClient(path=DB_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    total_samples = count_total_samples(json_paths_mc, json_paths_nonmc)
    processed = 0

    print(f"Total samples to process: {total_samples}")

    # ----------------------------
    # 객관식(MC) 데이터 처리
    # ----------------------------
    for path in json_paths_mc:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue

        samples = load_json(path)
        print(f"Processing {path}: {len(samples)} samples")

        for sample in samples:
            processed += 1
            percent = processed / total_samples * 100

            q_text = sample["question"]
            p_text = sample["positive"]
            n_texts = sample.get("negative", [])

            # ------- Encode Question -------
            print(f"[{processed}/{total_samples}] ({percent:.2f}%) Encoding question...")
            q_emb = model.encode(q_text).tolist()
            q_id = f"question_{processed}_q"

            documents.append(q_text)
            metadatas.append({"type": "question"})
            ids.append(q_id)
            embeddings.append(q_emb)

            # ------- Encode Positive -------
            print(f"[{processed}/{total_samples}] Encoding positive answer...")
            p_emb = model.encode(p_text).tolist()
            p_id = f"answer_{processed}_p"

            documents.append(p_text)
            metadatas.append({"type": "positive", "ref_id": q_id})
            ids.append(p_id)
            embeddings.append(p_emb)

            # ------- Encode Negatives -------
            if n_texts:
                print(f"[{processed}/{total_samples}] Encoding {len(n_texts)} negative samples...")
                n_embs = model.encode(n_texts)

                for i, neg_emb in enumerate(n_embs):
                    n_id = f"answer_{processed}_n{i}"
                    documents.append(n_texts[i])
                    metadatas.append({"type": "negative", "ref_id": q_id})
                    ids.append(n_id)
                    embeddings.append(neg_emb.tolist())

            # ------- Batch Flush -------
            if len(ids) >= BATCH_SIZE:
                flush_batch(collection, documents, metadatas, ids, embeddings)

        print(f"{path} finished.\n")

    # ----------------------------
    # 주관식(NON-MC) 데이터 처리
    # ----------------------------
    for path in json_paths_nonmc:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue

        samples = load_json(path)
        print(f"Processing {path}: {len(samples)} samples")

        for sample in samples:
            processed += 1
            percent = processed / total_samples * 100

            q_text = sample["question"]
            p_text = sample["positive"]

            print(f"[{processed}/{total_samples}] ({percent:.2f}%) Encoding question...")
            q_emb = model.encode(q_text).tolist()
            q_id = f"question_{processed}_q"

            documents.append(q_text)
            metadatas.append({"type": "question"})
            ids.append(q_id)
            embeddings.append(q_emb)

            print(f"[{processed}/{total_samples}] Encoding positive answer...")
            p_emb = model.encode(p_text).tolist()
            p_id = f"answer_{processed}_p"

            documents.append(p_text)
            metadatas.append({"type": "positive", "ref_id": q_id})
            ids.append(p_id)
            embeddings.append(p_emb)

            if len(ids) >= BATCH_SIZE:
                flush_batch(collection, documents, metadatas, ids, embeddings)

        print(f"{path} finished.\n")

    # ----------------------------
    # Final flush
    # ----------------------------
    print("Final flush...")
    flush_batch(collection, documents, metadatas, ids, embeddings)

    print("All data stored successfully in ChromaDB.")

if __name__ == "__main__":
    main()
