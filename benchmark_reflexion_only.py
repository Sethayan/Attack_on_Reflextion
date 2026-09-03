import csv
import json
import os
import sys
import time
from collections import defaultdict
from textwrap import dedent
from datetime import datetime, timezone
from typing import Dict, List

# Suppress CrewAI tracing prompt
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Crew
from dotenv import load_dotenv

from evaluator import evaluate
from reflexion_memory import ReflexionMemory
from trip_agents import TripAgents
from trip_tasks import TripTasks

load_dotenv()

#  Configuration

MAX_TRIALS = 3  

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

CSV_FILE = "benchmark_reflexion_only.csv"
JSON_FILE = "benchmark_reflexion_only.json"
TRACES_DIR = "traces_reflexion_only"

os.makedirs(TRACES_DIR, exist_ok=True)

CSV_COLUMNS = [
    "test_case", "trial", "overall_pass", "accuracy",
    "checks_passed", "checks_applicable",
    *CHECK_KEYS,
    "failure_reasons", "execution_time_s",
]




def save_trace(
    test_case_id: int,
    trial: int,
    crew_output: str,
    eval_result: dict,
    reflection: str = "",
):
    
    prefix = f"tc{test_case_id}_reflexion_trial{trial}"


    with open(os.path.join(TRACES_DIR, f"{prefix}_output.txt"), "w") as f:
        f.write(crew_output)

    
    trace = {
        "test_case_id": test_case_id,
        "trial": trial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_result": {k: v for k, v in eval_result.items()
                        if k not in ("judge_raw",)},
        "crew_output_length": len(crew_output),
        "reflection_preview": reflection[:1000] if reflection else "",
    }
    trace_path = os.path.join(TRACES_DIR, f"{prefix}_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    
    if reflection:
        with open(os.path.join(TRACES_DIR, f"{prefix}_reflection.txt"), "w") as f:
            f.write(reflection)

    print(f"Trace saved: {trace_path}")


def save_results_incremental(all_results: list):
   
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    with open(JSON_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)



def _build_task_description(tc: dict) -> str:
    return (
        f"Plan a trip from {tc['origin']} to one of "
        f"{tc['cities']} during {tc['date_range']}, "
        f"interests: {tc['interests']}"
    )


def _run_single_crew(test_case: dict, reflexion_context: str = "") -> str:
    """Run the trip planner crew once."""
    agents = TripAgents()
    tasks = TripTasks()

    city_selector = agents.city_selection_agent()
    local_expert = agents.local_expert()
    concierge = agents.travel_concierge()

    identify = tasks.identify_task(
        city_selector,
        test_case["origin"],
        test_case["cities"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=reflexion_context,
    )
    gather = tasks.gather_task(
        local_expert,
        test_case["origin"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=reflexion_context,
    )
    plan = tasks.plan_task(
        concierge,
        test_case["origin"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=reflexion_context,
    )

    crew = Crew(
        agents=[city_selector, local_expert, concierge],
        tasks=[identify, gather, plan],
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


#  Reflexion loop

def run_reflexion(test_case: dict, max_trials: int = MAX_TRIALS) -> List[dict]:
   
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

    
    print(f"\n    📌 Trial 0 (initial trajectory)...")

    
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
        print(f"    ❌ Trial 0 FAILED: {e}")
        output = f"[ERROR] Crew run failed: {e}"
    elapsed = time.time() - start

    
    eval_result = evaluate(task_input, output)
    print(f"    📊 Trial 0: accuracy={eval_result['accuracy']:.2f}, "
          f"pass={eval_result['overall_pass']}")

    trial_results.append({
        "test_case": test_case["id"],
        "trial": 0,
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
        print(f"    🪞 Generating self-reflection sr_0...")
        reflection_text = memory.reflect(task_desc, output, eval_result)
        memory.store(task_desc, output, reflection_text, eval_result)

    save_trace(test_case["id"], 0, output, eval_result, reflection_text)

    if eval_result["failure_reasons"]:
        for fr in eval_result["failure_reasons"]:
            print(f"       → {fr}")

   
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
                at this SAME task. Use them to fix your mistakes:

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
            print(f"    ✅ PASSED at trial {t}!")

        save_trace(test_case["id"], t, output, eval_result, reflection_text)

        if eval_result["failure_reasons"]:
            for fr in eval_result["failure_reasons"]:
                print(f"       → {fr}")

    return trial_results


#  Summary

def print_summary(all_results: list):
    """Print accuracy summary for Reflexion-only benchmark."""
    print(f"\n{'='*70}")
    print(f"  REFLEXION-ONLY SUMMARY")
    print(f"{'='*70}\n")


    by_tc = defaultdict(list)
    for r in all_results:
        by_tc[r["test_case"]].append(r)

    pass_count = sum(
        1 for trials in by_tc.values()
        if any(t["overall_pass"] for t in trials)
    )
    total_tc = len(by_tc)
    pass_at_1 = pass_count / total_tc if total_tc else 0

    print(f"  Pass@1 (eventual pass any trial): {pass_count}/{total_tc} = {pass_at_1:.4f}\n")

    
    print(f"  {'TC':<5} {'Trial':<7} {'Pass':<6} {'Accuracy':<10} "
          f"{'Time(s)':<8} {'Failures'}")
    print(f"  {'─'*5} {'─'*7} {'─'*6} {'─'*10} {'─'*8} {'─'*30}")

    for r in all_results:
        passed = "✅" if r["overall_pass"] else "❌"
        failures = r["failure_reasons"][:50] if r["failure_reasons"] else "—"
        print(f"  {r['test_case']:<5} {r['trial']:<7} {passed:<6} "
              f"{r['accuracy']:<10.4f} {r['execution_time_s']:<8} {failures}")

    
    print(f"\n  {'─'*60}")
    print(f"  ACCURACY PROGRESSION:\n")
    for tc_id, trials in sorted(by_tc.items()):
        accs = [f"{t['accuracy']:.2f}" for t in sorted(trials, key=lambda x: x['trial'])]
        best = max(t['accuracy'] for t in trials)
        print(f"  TC {tc_id}: {' → '.join(accs)}  (best: {best:.2f})")

    
    print(f"\n  {'─'*60}")
    print(f"  PER-CHECK PASS RATE (best trial per TC):\n")
    print(f"  {'Check':<20} {'Pass rate':<12}")
    print(f"  {'─'*20} {'─'*12}")

    for ck in CHECK_KEYS:
        pass_c = 0
        total_c = 0
        for tc_id, trials in by_tc.items():
            applicable = [t for t in trials if t.get(ck) not in ("not_applicable", None)]
            if applicable:
                total_c += 1
                if any(t.get(ck) is True for t in applicable):
                    pass_c += 1
        rate = f"{pass_c}/{total_c}" if total_c else "N/A"
        print(f"  {ck:<20} {rate:<12}")

    
    print(f"\n  {'─'*60}")
    times = [r["execution_time_s"] for r in all_results]
    avg_time = sum(times) / len(times) if times else 0
    total_time = sum(times)
    print(f"  Avg time per trial:  {avg_time:.1f}s")
    print(f"  Total benchmark time: {total_time:.1f}s")
    print()



if __name__ == "__main__":
    n_tc = len(TEST_CASES)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK: Reflexion-Only (Algorithm 1)                       ║")
    print(f"║  {n_tc} test cases × max {MAX_TRIALS} retry trials each                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    all_results = []

    for tc in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"  Test Case #{tc['id']}: {tc['origin']} → {tc['cities']}")
        print(f"  {tc['date_range']} | {tc['interests']}")
        print(f"{'='*70}")

        trial_results = run_reflexion(tc, max_trials=MAX_TRIALS)
        all_results.extend(trial_results)
        save_results_incremental(all_results)

    # Final save
    save_results_incremental(all_results)
    print(f"\n📄 CSV saved to {CSV_FILE}")
    print(f"📄 JSON saved to {JSON_FILE}")
    print(f"📁 Traces saved to {TRACES_DIR}/")

    print_summary(all_results)
