import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
from config import JD_PATH, CACHE_DIR
from logger import logger

def load_filtered_candidates(file_path):
    candidates = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates

def create_candidate_narrative(candidate):
    history = candidate.get('career_history', [])
    narrative_parts = []
    for role in history:
        desc = role.get('description', '')
        title = role.get('title', '')
        company = role.get('company', '')
        narrative_parts.append(f"At {company} as {title}: {desc}")
    return " ".join(narrative_parts)

def fast_tfidf_filter(jd_text, candidates, top_k=5000):
    logger.info(f"Running fast TF-IDF filter to reduce pool to {top_k}...")
    narratives = [create_candidate_narrative(c) for c in candidates]
    
    vectorizer = TfidfVectorizer(stop_words='english').fit(narratives + [jd_text])
    narrative_tfidf = vectorizer.transform(narratives)
    jd_tfidf = vectorizer.transform([jd_text])
    
    similarities = (narrative_tfidf * jd_tfidf.T).toarray().flatten()
    top_indices = np.argsort(similarities)[-top_k:]
    return [candidates[i] for i in top_indices], top_indices

def perform_semantic_search(jd_text, candidates, top_k=100, cache_dir=None):
    if cache_dir is None:
        cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    
    # Step 1: Fast Filter (Deterministic)
    filtered_candidates, indices = fast_tfidf_filter(jd_text, candidates, top_k=5000)
    
    # Step 2: Deep Semantic Search with Caching
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embed_dim = model.get_sentence_embedding_dimension()
    
    embeddings_path = os.path.join(cache_dir, "candidate_embeddings_5k.npy")
    temp_path = os.path.join(cache_dir, "candidate_embeddings_5k_tmp.npy")
    
    narratives = [create_candidate_narrative(c) for c in filtered_candidates]
    
    if os.path.exists(embeddings_path):
        logger.info(f"Loading cached embeddings from {embeddings_path}...")
        candidate_embeddings = np.load(embeddings_path)
    else:
        logger.info("Computing embeddings for 5000 candidates (with incremental save)...")
        if os.path.exists(temp_path):
            candidate_embeddings = np.load(temp_path)
            start_idx = len(candidate_embeddings)
            logger.info(f"Resuming from batch {start_idx // 32}...")
        else:
            candidate_embeddings = np.empty((0, embed_dim), dtype='float32')
            start_idx = 0

        batch_size = 32
        
        for i in tqdm(range(start_idx, len(narratives), batch_size)):
            batch = narratives[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False).astype('float32')
            candidate_embeddings = np.vstack([candidate_embeddings, batch_embeddings])
            
            # Save progress every 10 batches
            if (i // batch_size) % 10 == 0:
                np.save(temp_path, candidate_embeddings)
        
        np.save(embeddings_path, candidate_embeddings)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.info(f"Embeddings saved to {embeddings_path}")

    logger.info("Embedding Job Description...")
    jd_embedding = model.encode([jd_text])[0]
    
    candidate_embeddings = candidate_embeddings.astype('float32')
    faiss.normalize_L2(candidate_embeddings)
    
    jd_embedding = np.array([jd_embedding]).astype('float32')
    faiss.normalize_L2(jd_embedding)
    
    dimension = candidate_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(candidate_embeddings)
    
    distances, indices = index.search(jd_embedding, top_k)
    
    shortlisted = []
    for idx in indices[0]:
        if idx != -1 and idx < len(filtered_candidates):
            shortlisted.append(filtered_candidates[idx])
            
    return shortlisted

def run_semantic_search_stage(jd_text, filtered_candidates, output_dir, cache_dir=None):
    """Run Stage 1: Semantic Search and save to output_dir/2_shortlisted_candidates.jsonl"""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Semantic Search (Hybrid)")
    logger.info("=" * 60)
    
    output_path = os.path.join(output_dir, "2_shortlisted_candidates.jsonl")
    
    if os.path.exists(output_path):
        logger.info(f"Found existing {output_path}, skipping...")
        with open(output_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    logger.info("Running hybrid semantic search...")
    shortlisted = perform_semantic_search(jd_text, filtered_candidates, top_k=100, cache_dir=cache_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for cand in shortlisted:
            f.write(json.dumps(cand) + '\n')
    logger.info(f"Shortlisted {len(shortlisted)} candidates saved to {output_path}")
    return shortlisted

if __name__ == "__main__":
    from run_manager import create_run_folder
    run_dir = create_run_folder()
    
    if not os.path.exists(JD_PATH):
        logger.error(f"JD file not found at {JD_PATH}")
        exit(1)

    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    
    logger.info(f"Loading filtered candidates...")
    candidates = load_filtered_candidates(JD_PATH.parent / "filtered_candidates.jsonl")
    
    logger.info(f"Performing hybrid semantic search to retrieve top 100 candidates...")
    shortlisted_candidates = perform_semantic_search(jd_text, candidates, top_k=100)
    
    logger.info(f"Shortlisted {len(shortlisted_candidates)} candidates.")
    
    output_path = os.path.join(run_dir, "2_shortlisted_candidates.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for cand in shortlisted_candidates:
            f.write(json.dumps(cand) + '\n')
    logger.info(f"Shortlisted candidates saved to {output_path}")
