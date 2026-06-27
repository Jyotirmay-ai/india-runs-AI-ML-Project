import json
import pandas as pd
from datetime import datetime, timedelta
import os
from config import CANDIDATES_PATH
from logger import logger

def load_candidates(file_path):
    candidates = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates

def prefilter_candidates(candidates, active_days=180, min_response_rate=0.05):
    """
    Filters candidates based on behavioral signals.
    - last_active_date must be within the last X days.
    - recruiter_response_rate must be at least X.
    """
    # Use current date as per system prompt (Mon Jun 22 2026)
    today = datetime(2026, 6, 22)
    cutoff_date = today - timedelta(days=active_days)
    
    filtered = []
    for cand in candidates:
        signals = cand.get('redrob_signals', {})
        last_active_str = signals.get('last_active_date')
        response_rate = signals.get('recruiter_response_rate', 0)
        
        if not last_active_str:
            continue
            
        last_active = datetime.fromisoformat(last_active_str.replace('Z', '+00:00')).replace(tzinfo=None)
        
        if last_active >= cutoff_date and response_rate >= min_response_rate:
            filtered.append(cand)
            
    return filtered

def run_prefilter_stage(dataset_path, output_dir):
    """Run Stage 0: Behavioral Pre-filtering and save to output_dir/1_filtered_candidates.jsonl"""
    logger.info("=" * 60)
    logger.info("STAGE 0: Behavioral Pre-filtering")
    logger.info("=" * 60)
    
    output_path = os.path.join(output_dir, "1_filtered_candidates.jsonl")
    
    if os.path.exists(output_path):
        logger.info(f"Found existing {output_path}, skipping...")
        with open(output_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    logger.info(f"Loading candidates from {dataset_path}...")
    all_candidates = load_candidates(dataset_path)
    logger.info(f"Total candidates: {len(all_candidates)}")
    
    filtered = prefilter_candidates(all_candidates)
    logger.info(f"Candidates after pre-filtering: {len(filtered)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for cand in filtered:
            f.write(json.dumps(cand) + '\n')
    logger.info(f"Saved to {output_path}")
    return filtered

if __name__ == "__main__":
    from run_manager import create_run_folder
    run_dir = create_run_folder()
    run_prefilter_stage(CANDIDATES_PATH, run_dir)
