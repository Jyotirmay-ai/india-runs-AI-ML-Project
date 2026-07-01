import json
import os
import shutil
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv
from config import JD_PATH, PROGRESS_FILE
from logger import logger

load_dotenv()

# Initialize Groq client (requires GROQ_API_KEY environment variable)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def clear_progress_checkpoint():
    """Clear the persistent progress checkpoint (use when changing dataset/JD)."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        logger.info(f"Cleared progress checkpoint: {PROGRESS_FILE}")
    else:
        logger.info("No progress checkpoint to clear")


def load_shortlisted_candidates(file_path):
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

def build_llm_prompt(jd_text, candidate):
    narrative = create_candidate_narrative(candidate)
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    skill_names = [s['name'] for s in skills]
    
    return f"""You are an expert technical recruiter for Redrob AI, evaluating candidates for a **Senior AI Engineer — Founding Team** role.

**Job Description:**
{jd_text}

**Candidate Profile:**
- Name: {profile.get('anonymized_name', 'N/A')}
- Current Title: {profile.get('current_title', 'N/A')} at {profile.get('current_company', 'N/A')}
- Years Experience: {profile.get('years_of_experience', 'N/A')}
- Location: {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}
- Headline: {profile.get('headline', 'N/A')}
- Summary: {profile.get('summary', 'N/A')}
- Skills: {', '.join(skill_names[:20])}

**Career History (Detailed):**
{narrative}

**Evaluation Criteria (from JD):**
1. **Production Embeddings/Retrieval**: Has the candidate built and deployed embedding-based retrieval systems (sentence-transformers, BGE, E5, OpenAI embeddings) to real users? Handled embedding drift, index refresh, retrieval-quality regression?
2. **Vector Database / Hybrid Search**: Production experience with Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar. Operational experience matters.
3. **Strong Python & Code Quality**: Demonstrable production-grade Python.
4. **Ranking Evaluation Frameworks**: Designed evaluation for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).
5. **Pre-LLM ML Depth**: Understood retrieval/ranking before it was fashionable. Not just recent LangChain/OpenAI wrappers.
6. **Product Company Experience**: Product companies preferred over pure consulting/services (TCS, Infosys, Wipro, etc. are disqualifiers unless prior product experience).
7. **Shipper Mindset**: Willing to ship v2 ranker in weeks, not just research. "Tilt toward shipper."
8. **Behavioral Signals**: Active, responsive candidate (handled in pre-filter).

**Output Format (JSON only):**
{{
  "score": <0-100 integer>,
  "breakdown": {{
    "hard_skills": <0-100>,
    "experience_relevance": <0-100>,
    "behavioral_fit": <0-100>,
    "shipper_mindset": <0-100>
  }},
  "reasoning": "<2-3 sentences explaining the score, referencing specific career history evidence>",
  "red_flags": "<any disqualifiers found: pure consulting, only recent LLM wrappers, no production retrieval, title-chaser, etc.>"
}}"""

def evaluate_candidate(jd_text, candidate):
    prompt = build_llm_prompt(jd_text, candidate)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Best free model on Groq
            messages=[
                {"role": "system", "content": "You are a rigorous technical recruiter. Output ONLY valid JSON. No markdown, no extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result["candidate_id"] = candidate["candidate_id"]
        result["name"] = candidate["profile"].get("anonymized_name", "")
        return result
    except Exception as e:
        logger.error(f"Error evaluating {candidate['candidate_id']}: {e}")
        return {
            "candidate_id": candidate["candidate_id"],
            "name": candidate["profile"].get("anonymized_name", ""),
            "score": 0,
            "breakdown": {"hard_skills": 0, "experience_relevance": 0, "behavioral_fit": 0, "shipper_mindset": 0},
            "reasoning": f"API Error: {e}",
            "red_flags": "Evaluation failed"
        }

def run_llm_ranker_stage(jd_text, shortlisted_candidates, output_dir):
    """Run Stage 2: LLM Recruiter Scoring and save numbered files to output_dir"""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: LLM Recruiter Scoring")
    logger.info("=" * 60)
    
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not set. Set it in .env or environment.")
        return []
    
    ranked_path = os.path.join(output_dir, "3_ranked_candidates.jsonl")
    progress_path = str(PROGRESS_FILE)  # Shared progress file across runs
    
    if os.path.exists(ranked_path):
        logger.info(f"Found existing {ranked_path}, loading...")
        with open(ranked_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    completed_ids = set()
    results = []
    if os.path.exists(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                completed_ids.add(r["candidate_id"])
        logger.info(f"Resuming from {len(completed_ids)} completed (from shared progress file)...")
    
    remaining = [c for c in shortlisted_candidates if c["candidate_id"] not in completed_ids]
    logger.info(f"Evaluating {len(remaining)} candidates via Groq...")
    
    for cand in tqdm(remaining):
        result = evaluate_candidate(jd_text, cand)
        results.append(result)
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 3_ranked_candidates.jsonl - full ranked results
    with open(ranked_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Copy progress file to run folder for record-keeping
    import shutil
    run_progress_path = os.path.join(output_dir, "4_ranking_progress.jsonl")
    shutil.copy2(PROGRESS_FILE, run_progress_path)
    
    logger.info(f"Ranking complete. Top 5:")
    for i, r in enumerate(results[:5]):
        logger.info(f"  {i+1}. {r['name']} ({r['candidate_id']}) - Score: {r['score']}")
    
    logger.info(f"Deliverables saved to {output_dir}:")
    logger.info(f"  - 3_ranked_candidates.jsonl ({len(results)} candidates)")
    logger.info(f"  - 4_ranking_progress.jsonl (checkpoint copy)")
    logger.info(f"Progress checkpoint: {PROGRESS_FILE} (persists across runs)")
    return results

if __name__ == "__main__":
    from run_manager import create_run_folder
    run_dir = create_run_folder()
    
    if not os.getenv("GROQ_API_KEY"):
        logger.error("Set GROQ_API_KEY environment variable first.")
        logger.info("Get free key at: https://console.groq.com/keys")
        exit(1)
    
    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    
    # We need candidates to run this standalone, so we load from a default output location
    # but this is mostly used via main.py
    try:
        shortlisted_path = os.path.join(run_dir, "2_shortlisted_candidates.jsonl")
        candidates = load_shortlisted_candidates(shortlisted_path)
    except FileNotFoundError:
        logger.error("Shortlisted candidates file not found. Please run the full pipeline via main.py")
        exit(1)
        
    run_llm_ranker_stage(jd_text, candidates, run_dir)
