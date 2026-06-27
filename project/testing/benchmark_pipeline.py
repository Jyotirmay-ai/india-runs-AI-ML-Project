"""
Pipeline Benchmarking Tool
This script measures the execution time for each stage of the AI Candidate Ranking System.
"""
import os
import sys
import time
from pathlib import Path

# Add src to path so we can import the pipeline stages
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import validate_setup, CANDIDATES_PATH, JD_PATH
from src.prefilter import run_prefilter_stage
from src.semantic_search import run_semantic_search_stage
from src.llm_ranker import run_llm_ranker_stage
from src.run_manager import create_run_folder

def benchmark_pipeline():
    # Ensure setup is valid
    try:
        validate_setup()
    except FileNotFoundError as e:
        print(f"Setup Error: {e}")
        return

    print("Starting Pipeline Benchmark...")
    print("=" * 60)
    
    # Create a dedicated folder for benchmark results
    run_dir = create_run_folder()
    print(f"Benchmark Run Folder: {run_dir}")
    
    # Read JD text once
    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    total_start = time.perf_counter()
    stage_times = {}

    # --- Stage 0 ---
    print("\n[BMRK] Timing Stage 0: Behavioral Pre-filtering...")
    s0_start = time.perf_counter()
    filtered = run_prefilter_stage(CANDIDATES_PATH, run_dir)
    s0_end = time.perf_counter()
    stage_times['Stage 0'] = s0_end - s0_start

    # --- Stage 1 ---
    print("\n[BMRK] Timing Stage 1: Semantic Search...")
    s1_start = time.perf_counter()
    shortlisted = run_semantic_search_stage(jd_text, filtered, run_dir)
    s1_end = time.perf_counter()
    stage_times['Stage 1'] = s1_end - s1_start

    # --- Stage 2 ---
    print("\n[BMRK] Timing Stage 2: LLM Scoring...")
    s2_start = time.perf_counter()
    ranked = run_llm_ranker_stage(jd_text, shortlisted, run_dir)
    s2_end = time.perf_counter()
    stage_times['Stage 2'] = s2_end - s2_start

    total_end = time.perf_counter()
    total_duration = total_end - total_start

    # --- Report ---
    print("\n" + "=" * 60)
    print("PIPELINE BENCHMARK REPORT")
    print("=" * 60)
    for stage, duration in stage_times.items():
        print(f"{stage: <20} : {duration:>10.4f} seconds")
    print("-" * 60)
    print(f"TOTAL TIME           : {total_duration:>10.4f} seconds")
    print("=" * 60)
    print(f"Detailed results saved in: {run_dir}")

if __name__ == "__main__":
    benchmark_pipeline()