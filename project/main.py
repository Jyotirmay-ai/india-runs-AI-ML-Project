"""
AI Candidate Ranking System - Main Pipeline Orchestrator
Run: python main.py
"""
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import validate_setup, CANDIDATES_PATH, JD_PATH
from src.prefilter import run_prefilter_stage
from src.semantic_search import run_semantic_search_stage
from src.llm_ranker import run_llm_ranker_stage
from src.run_manager import create_run_folder
from src.logger import logger
from src.exporter import merge_candidate_data, export_ranked_candidates


def get_top_candidates_count():
    """Ask user how many top candidates they want in the final spreadsheet."""
    print("\n" + "=" * 60)
    print("AI CANDIDATE RANKING SYSTEM")
    print("=" * 60)
    print("The pipeline will rank 100 candidates via LLM evaluation.")
    print("How many top candidates do you want in the final CSV/Excel?")
    print("(Press Enter for default: 60)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("Enter number of top candidates: ").strip()
            if not user_input:
                return 60  # default
            count = int(user_input)
            if count <= 0:
                print("Please enter a positive number.")
                continue
            if count > 100:
                print("Maximum is 100 (total ranked candidates).")
                continue
            return count
        except ValueError:
            print("Please enter a valid number.")


def main():
    try:
        # Validate setup once at startup
        validate_setup()
        
        # Ask user how many top candidates they want
        top_n = get_top_candidates_count()
        
        logger.info("AI CANDIDATE RANKING SYSTEM - FULL PIPELINE")
        logger.info("=" * 60)
        
        # Create timestamped run folder
        run_dir = create_run_folder()
        logger.info(f"Run folder: {run_dir}")
        logger.info("=" * 60)
        
        # Timing dictionary to track each stage
        stage_times = {}
        total_start = time.perf_counter()
        
        # Stage 0: Behavioral Pre-filtering
        logger.info("Starting Stage 0: Behavioral Pre-filtering")
        s0_start = time.perf_counter()
        filtered = run_prefilter_stage(CANDIDATES_PATH, run_dir)
        stage_times['Stage 0 (Pre-filtering)'] = time.perf_counter() - s0_start
        logger.info(f"Stage 0 completed in {stage_times['Stage 0 (Pre-filtering)']:.2f}s. Filtered: {len(filtered)} candidates")
        
        # Stage 1: Semantic Search
        with open(JD_PATH, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        
        logger.info("Starting Stage 1: Semantic Search")
        s1_start = time.perf_counter()
        shortlisted = run_semantic_search_stage(jd_text, filtered, run_dir)
        stage_times['Stage 1 (Semantic Search)'] = time.perf_counter() - s1_start
        logger.info(f"Stage 1 completed in {stage_times['Stage 1 (Semantic Search)']:.2f}s. Shortlisted: {len(shortlisted)} candidates")
        
        # Stage 2: LLM Recruiter Scoring
        logger.info("Starting Stage 2: LLM Recruiter Scoring")
        s2_start = time.perf_counter()
        ranked = run_llm_ranker_stage(jd_text, shortlisted, run_dir)
        stage_times['Stage 2 (LLM Scoring)'] = time.perf_counter() - s2_start
        logger.info(f"Stage 2 completed in {stage_times['Stage 2 (LLM Scoring)']:.2f}s. Ranked: {len(ranked)} candidates")
        
        # Export to CSV/XLSX (merges full profiles with evaluation results)
        logger.info("Generating CSV/XLSX exports...")
        merged = merge_candidate_data(shortlisted, ranked)
        # Sort by LLM score (descending) and take top N
        merged.sort(key=lambda x: x["evaluation"].get("score", 0), reverse=True)
        merged_top = merged[:top_n]
        export_ranked_candidates(merged_top, run_dir)
        stage_times['Export (CSV/XLSX)'] = time.perf_counter() - s2_start - stage_times['Stage 2 (LLM Scoring)']
        
        total_duration = time.perf_counter() - total_start
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"All deliverables in: {run_dir}")
        logger.info(f"  - 1_filtered_candidates.jsonl ({len(filtered)} candidates)")
        logger.info(f"  - 2_shortlisted_candidates.jsonl ({len(shortlisted)} candidates)")
        logger.info(f"  - 3_ranked_candidates.jsonl ({len(ranked)} candidates)")
        logger.info(f"  - 4_ranking_progress.jsonl (checkpoint)")
        logger.info(f"  - 5_ranked_candidates.csv  (top {len(merged_top)} candidates, spreadsheet)")
        logger.info(f"  - 5_ranked_candidates.xlsx (top {len(merged_top)} candidates, spreadsheet)")
        
        logger.info("\n" + "-" * 30)
        logger.info("EXECUTION TIME REPORT")
        logger.info("-" * 30)
        for stage, duration in stage_times.items():
            logger.info(f"{stage: <25} : {duration:>10.4f} seconds")
        logger.info("-" * 30)
        logger.info(f"TOTAL TIME              : {total_duration:>10.4f} seconds")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()