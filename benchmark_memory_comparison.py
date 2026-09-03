import csv
import json
import os
import sys
import time
from collections import defaultdict
from textwrap import dedent
from typing import Dict, List, Tuple

# Suppress CrewAI tracing prompt (cosmetic — not an error)
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Crew
from dotenv import load_dotenv

from evaluator import evaluate
from reflexion_memory import ReflexionMemory
from trip_agents import TripAgents
from trip_tasks import TripTasks

load_dotenv()



MAX_TRIALS = 3          
NUM_RUNS = 1            
                    

TEST_CASES = [
    {
        "id": 1,
        "origin": "Mumbai",
        "cities": "Paris, Rome, Barcelona",
        "date_range": "June 1-7, 2026",
        "interests": "art, food, history",
    },
    {
        "id": 2,
        "origin": "London",
        "cities": "Lisbon, Athens, Istanbul",
        "date_range": "September 1-5, 2026",
        "interests": "beaches, nightlife",
    },
]

CHECK_KEYS = [
    "budget_ok", "days_ok", "no_duplicates",
    "constraints_ok", "feasible", "no_hallucination",
]

CSV_FILE = "benchmark_reflexion.csv"
JSON_FILE = "benchmark_reflexion_results.json"
TRACES_DIR = "traces"




os.makedirs(TRACES_DIR, exist_ok=True)


def save_trace(
    test_case_id: int,
    condition: str,
    trial: int,
    crew_output: str,
    eval_result: dict,
    reflection: str = "",
):
    """Save a complete trace for one trial to disk.

    Creates a JSON file: traces/tc{id}_{condition}_trial{trial}.json
    and a plain-text file with the raw crew output.
    """
    prefix = f"tc{test_case_id}_{condition}_trial{trial}"

    # Save raw crew output
    output_path = os.path.join(TRACES_DIR, f"{prefix}_output.txt")
    with open(output_path, "w") as f:
        f.write(crew_output)

    # Save structured trace (eval + metadata)
    trace = {
        "test_case_id": test_case_id,
        "condition": condition,
        "trial": trial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_result": {k: v for k, v in eval_result.items()
                        if k not in ("judge_raw",)},
        "crew_output_length": len(crew_output),
        "crew_output_file": output_path,
        "reflection_preview": reflection[:1000] if reflection else "",
    }
    trace_path = os.path.join(TRACES_DIR, f"{prefix}_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    # Save reflection if present
    if reflection:
        ref_path = os.path.join(TRACES_DIR, f"{prefix}_reflection.txt")
        with open(ref_path, "w") as f:
            f.write(reflection)

    print(f"    💾 Trace saved: {trace_path}")


def save_results_incremental(all_results: list):
    """Save results CSV and JSON incrementally (called after each run)."""
    # CSV
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    # JSON
    with open(JSON_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


from datetime import datetime, timezone




def _build_task_description(tc: dict) -> str:
    return (
        f"Plan a trip from {tc['origin']} to one of "
        f"{tc['cities']} during {tc['date_range']}, "
        f"interests: {tc['interests']}"
    )


def _run_single_crew(
    test_case: dict,
    reflexion_context: str = "",
) -> str:
    """Run the trip planner crew once (Actor generates trajectory τ_t)."""
    agents = TripAgents()
    tasks = TripTasks()

    city_selector_agent = agents.city_selection_agent()
    local_expert_agent = agents.local_expert()
    travel_concierge_agent = agents.travel_concierge()

    identify_task = tasks.identify_task(
        city_selector_agent,
        test_case["origin"],
        test_case["cities"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=reflexion_context,
    )
    gather_task = tasks.gather_task(
        local_expert_agent,
        test_case["origin"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=reflexion_context,
    )
    plan_task = tasks.plan_task(
        travel_concierge_agent,
        test_case["origin"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=reflexion_context,
    )

    crew = Crew(
        agents=[city_selector_agent, local_expert_agent, travel_concierge_agent],
        tasks=[identify_task, gather_task, plan_task],
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)




def run_baseline(test_case: dict) -> dict:
    """Baseline: single-shot crew run, no memory, no retries.

    Returns a result dict with eval scores and timing.
    """
    task_input = {
        "origin": test_case["origin"],
        "cities": test_case["cities"],
        "date_range": test_case["date_range"],
        "interests": test_case["interests"],
    }

    start = time.time()
    try:
        output = _run_single_crew(test_case)
    except Exception as e:
        print(f"     Crew run FAILED: {e}")
        output = f"[ERROR] Crew run failed: {e}"
    elapsed = time.time() - start

    eval_result = evaluate(task_input, output)

    # Save trace immediately
    save_trace(
        test_case_id=test_case["id"],
        condition="baseline",
        trial=1,
        crew_output=output,
        eval_result=eval_result,
    )

    return {
        "test_case": test_case["id"],
        "condition": "baseline",
        "trial": 1,
        "overall_pass": eval_result["overall_pass"],
        "accuracy": eval_result["accuracy"],
        "checks_passed": eval_result["checks_passed"],
        "checks_applicable": eval_result["checks_applicable"],
        "failure_reasons": "; ".join(eval_result["failure_reasons"]),
        "execution_time_s": round(elapsed, 1),
        **{ck: eval_result.get(ck) for ck in CHECK_KEYS},
    }


def run_reflexion(test_case: dict, max_trials: int = MAX_TRIALS) -> List[dict]:
    """Reflexion (Algorithm 1): retry loop with self-reflection memory.

    Pseudocode from the paper:
        Generate initial trajectory τ_0 using π_θ
        Evaluate τ_0 using M_e
        Generate initial self-reflection sr_0 using M_sr
        Set mem ← [sr_0]
        Set t = 0
        while M_e not pass AND t < max_trials do
            Generate τ_t using π_θ  (with mem as context)
            Evaluate τ_t using M_e
            Generate sr_t using M_sr (with evaluator feedback)
            Append sr_t to mem
            t += 1
        end while

    Returns a list of result dicts (one per trial).
    """
    task_input = {
        "origin": test_case["origin"],
        "cities": test_case["cities"],
        "date_range": test_case["date_range"],
        "interests": test_case["interests"],
    }
    task_desc = _build_task_description(test_case)

    memory = ReflexionMemory()
    memory.clear_session()
    trial_results = []

    # ── Step 1: Generate initial trajectory τ_0 ─────────────────────
    print(f"     Trial 0 (initial trajectory)...")

    # Retrieve any long-term reflections from prior benchmark runs
    past_reflections = memory.retrieve_relevant(task_desc)
    reflexion_context = ""
    if past_reflections:
        lessons = "\n---\n".join(past_reflections)
        reflexion_context = dedent(f"""\

            === LESSONS FROM PREVIOUS RUNS ===
            {lessons}
            === END OF LESSONS ===

        """)

    start = time.time()
    try:
        output = _run_single_crew(test_case, reflexion_context=reflexion_context)
    except Exception as e:
        print(f"     Trial 0 FAILED: {e}")
        output = f"[ERROR] Crew run failed: {e}"
    elapsed = time.time() - start

    
    eval_result = evaluate(task_input, output)
    print(f"    Trial 0: accuracy={eval_result['accuracy']:.2f}, "
          f"pass={eval_result['overall_pass']}")

    trial_results.append({
        "test_case": test_case["id"],
        "condition": "reflexion",
        "trial": 0,
        "overall_pass": eval_result["overall_pass"],
        "accuracy": eval_result["accuracy"],
        "checks_passed": eval_result["checks_passed"],
        "checks_applicable": eval_result["checks_applicable"],
        "failure_reasons": "; ".join(eval_result["failure_reasons"]),
        "execution_time_s": round(elapsed, 1),
        **{ck: eval_result.get(ck) for ck in CHECK_KEYS},
    })

    
    print(f"    🪞 Generating self-reflection sr_0...")
    reflection_text = memory.reflect(task_desc, output, eval_result)
    memory.store(task_desc, output, reflection_text, eval_result)

    # Save trace immediately
    save_trace(
        test_case_id=test_case["id"],
        condition="reflexion",
        trial=0,
        crew_output=output,
        eval_result=eval_result,
        reflection=reflection_text,
    )

    
    t = 0
    while not eval_result["overall_pass"] and t < max_trials:
        t += 1
        print(f"\n    📌 Trial {t}/{max_trials} (reflexion retry)...")

    
        session_refs = memory.get_session_reflections()
        if session_refs:
            lessons = "\n---\n".join(session_refs)
            reflexion_context = dedent(f"""\

                === SELF-REFLECTIONS FROM PREVIOUS TRIALS ===
                The following are your own self-critiques from prior attempts
                at this SAME task.  Use them to fix your mistakes:

                {lessons}

                === END OF SELF-REFLECTIONS ===

            """)
        else:
            reflexion_context = ""

        
        long_term = memory.retrieve_relevant(task_desc)
        if long_term:
            lt_text = "\n---\n".join(long_term)
            reflexion_context += dedent(f"""\

                === LESSONS FROM EXPERIENCE (LONG-TERM MEMORY) ===
                {lt_text}
                === END OF LESSONS ===

            """)

        
        start = time.time()
        try:
            output = _run_single_crew(test_case, reflexion_context=reflexion_context)
        except Exception as e:
            print(f"    ❌ Trial {t} FAILED: {e}")
            output = f"[ERROR] Crew run failed: {e}"
        elapsed = time.time() - start

        
        eval_result = evaluate(task_input, output)
        print(f"    📊 Trial {t}: accuracy={eval_result['accuracy']:.2f}, "
              f"pass={eval_result['overall_pass']}")

        trial_results.append({
            "test_case": test_case["id"],
            "condition": "reflexion",
            "trial": t,
            "overall_pass": eval_result["overall_pass"],
            "accuracy": eval_result["accuracy"],
            "checks_passed": eval_result["checks_passed"],
            "checks_applicable": eval_result["checks_applicable"],
            "failure_reasons": "; ".join(eval_result["failure_reasons"]),
            "execution_time_s": round(elapsed, 1),
            **{ck: eval_result.get(ck) for ck in CHECK_KEYS},
        })

        
        reflection_text = ""
        if not eval_result["overall_pass"]:
            print(f"    🪞 Generating self-reflection sr_{t}...")
            reflection_text = memory.reflect(task_desc, output, eval_result)
            memory.store(task_desc, output, reflection_text, eval_result)
        else:
            print(f"    ✅ PASSED at trial {t}! No further reflection needed.")

        # Save trace immediately
        save_trace(
            test_case_id=test_case["id"],
            condition="reflexion",
            trial=t,
            crew_output=output,
            eval_result=eval_result,
            reflection=reflection_text,
        )

        if eval_result["failure_reasons"]:
            for fr in eval_result["failure_reasons"]:
                print(f"       → {fr}")

    return trial_results




CSV_COLUMNS = [
    "test_case", "condition", "trial", "overall_pass", "accuracy",
    "checks_passed", "checks_applicable",
    *CHECK_KEYS,
    "failure_reasons", "execution_time_s",
]


def run_benchmark():
    """Run all test cases under baseline and reflexion, report accuracy."""
    all_results = []


    print(f"\n{'='*70}")
    print(f"  CONDITION: BASELINE (single-shot, no memory)")
    print(f"{'='*70}\n")

    for tc in TEST_CASES:
        print(f"\n--- Baseline: Test #{tc['id']} ---")
        print(f"    {tc['origin']} → {tc['cities']} | "
              f"{tc['date_range']} | {tc['interests']}")

        result = run_baseline(tc)
        all_results.append(result)
        save_results_incremental(all_results)  # save after each run

        status = "✅ PASS" if result["overall_pass"] else "❌ FAIL"
        print(f"    ⏱  {result['execution_time_s']}s | "
              f"Accuracy: {result['accuracy']:.2f} | {status}")
        if result["failure_reasons"]:
            for fr in result["failure_reasons"].split("; "):
                print(f"       → {fr}")

    #
    print(f"\n{'='*70}")
    print(f"  CONDITION: REFLEXION (retry loop, max {MAX_TRIALS} trials)")
    print(f"{'='*70}\n")

    for tc in TEST_CASES:
        print(f"\n--- Reflexion: Test #{tc['id']} ---")
        print(f"    {tc['origin']} → {tc['cities']} | "
              f"{tc['date_range']} | {tc['interests']}")

        trial_results = run_reflexion(tc, max_trials=MAX_TRIALS)
        all_results.extend(trial_results)
        save_results_incremental(all_results)  # save after each test case

    
    save_results_incremental(all_results)
    print(f"\n CSV saved to {CSV_FILE}")
    print(f" JSON saved to {JSON_FILE}")
    print(f" Traces saved to {TRACES_DIR}/")

    
    print_accuracy_summary(all_results)


def print_accuracy_summary(results: list):
    """Print Pass@1 accuracy summary matching Reflexion paper Table 4 style."""
    print(f"\n{'='*70}")
    print("  ACCURACY SUMMARY (Reflexion Paper Style)")
    print(f"{'='*70}\n")

   
    baseline_results = [r for r in results if r["condition"] == "baseline"]
    baseline_pass = sum(1 for r in baseline_results if r["overall_pass"])
    baseline_total = len(baseline_results)
    baseline_pass_at_1 = baseline_pass / baseline_total if baseline_total else 0


    reflexion_results = [r for r in results if r["condition"] == "reflexion"]
    reflexion_by_tc = defaultdict(list)
    for r in reflexion_results:
        reflexion_by_tc[r["test_case"]].append(r)

    reflexion_pass = sum(
        1 for trials in reflexion_by_tc.values()
        if any(t["overall_pass"] for t in trials)
    )
    reflexion_total = len(reflexion_by_tc)
    reflexion_pass_at_1 = reflexion_pass / reflexion_total if reflexion_total else 0

   
    baseline_accs = [r["accuracy"] for r in baseline_results]
    baseline_avg_acc = sum(baseline_accs) / len(baseline_accs) if baseline_accs else 0

    
    reflexion_best_accs = []
    for tc_id, trials in reflexion_by_tc.items():
        best_acc = max(t["accuracy"] for t in trials)
        reflexion_best_accs.append(best_acc)
    reflexion_avg_acc = (
        sum(reflexion_best_accs) / len(reflexion_best_accs)
        if reflexion_best_accs else 0
    )


    print(f"  {'Approach':<15} {'Pass@1 accuracy':<25} {'Avg accuracy score':<20}")
    print(f"  {'-'*15} {'-'*25} {'-'*20}")
    print(f"  {'Baseline':<15} {baseline_pass_at_1:<25.4f} {baseline_avg_acc:<20.4f}")
    print(f"  {'Reflexion':<15} {reflexion_pass_at_1:<25.4f} {reflexion_avg_acc:<20.4f}")


    delta_pass = reflexion_pass_at_1 - baseline_pass_at_1
    delta_acc = reflexion_avg_acc - baseline_avg_acc
    print()
    print(f"  Δ Pass@1:   {delta_pass:+.4f}")
    print(f"  Δ Accuracy: {delta_acc:+.4f}")

    
    print(f"\n  {'─'*60}")
    print(f"  PER TEST CASE BREAKDOWN:\n")

    print(f"  {'TC':<5} {'Condition':<12} {'Trial':<7} {'Pass':<6} "
          f"{'Accuracy':<10} {'Time(s)':<8} {'Failures'}")
    print(f"  {'─'*5} {'─'*12} {'─'*7} {'─'*6} {'─'*10} {'─'*8} {'─'*30}")

    for r in results:
        passed = "Yes" if r["overall_pass"] else "No"
        failures = r["failure_reasons"][:50] if r["failure_reasons"] else "—"
        print(f"  {r['test_case']:<5} {r['condition']:<12} {r['trial']:<7} "
              f"{passed:<6} {r['accuracy']:<10.4f} "
              f"{r['execution_time_s']:<8} {failures}")

    
    print(f"\n  {'─'*60}")
    print(f"  PER-CHECK PASS RATE:\n")
    print(f"  {'Check':<20} {'Baseline':<12} {'Reflexion (best)':<18}")
    print(f"  {'─'*20} {'─'*12} {'─'*18}")

    for ck in CHECK_KEYS:
        # Baseline
        bl_applicable = [r for r in baseline_results if r.get(ck) not in ("not_applicable", None)]
        bl_pass = sum(1 for r in bl_applicable if r.get(ck) is True)
        bl_rate = f"{bl_pass}/{len(bl_applicable)}" if bl_applicable else "N/A"

        
        rx_pass = 0
        rx_total = 0
        for tc_id, trials in reflexion_by_tc.items():
            applicable = [t for t in trials if t.get(ck) not in ("not_applicable", None)]
            if applicable:
                rx_total += 1
                if any(t.get(ck) is True for t in applicable):
                    rx_pass += 1
        rx_rate = f"{rx_pass}/{rx_total}" if rx_total else "N/A"

        print(f"  {ck:<20} {bl_rate:<12} {rx_rate:<18}")


    print(f"\n  {'─'*60}")
    bl_times = [r["execution_time_s"] for r in baseline_results]
    rx_times = [r["execution_time_s"] for r in reflexion_results]
    bl_avg = sum(bl_times) / len(bl_times) if bl_times else 0
    rx_avg = sum(rx_times) / len(rx_times) if rx_times else 0
    rx_total_time = sum(rx_times)

    print(f"  Baseline avg time per test:  {bl_avg:.1f}s")
    print(f"  Reflexion avg time per trial: {rx_avg:.1f}s")
    print(f"  Reflexion total time:         {rx_total_time:.1f}s")
    print()


if __name__ == "__main__":
    n_tc = len(TEST_CASES)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK: Reflexion Algorithm 1 (Shinn et al., 2023)         ║")
    print(f"║  {n_tc} test cases × baseline + reflexion (max {MAX_TRIALS} trials)            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    run_benchmark()
