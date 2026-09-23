import csv
import json
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
from typing import Dict, List

# Suppress CrewAI tracing prompt
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Crew, Process
from dotenv import load_dotenv

from evaluator import evaluate, evaluate_episode
from reflexion_memory import ReflexionMemory
from trip_agents import TripAgents
from trip_tasks import TripTasks

load_dotenv()


# ── Reflexion granularity (Change 2) ────────────────────────────────

class ReflexionGranularity(Enum):
    PER_TRIAL = "per_trial"       # existing behavior — reflect once after full plan
    PER_EPISODE = "per_episode"   # new — reflect after every individual agent step

REFLEXION_GRANULARITY = ReflexionGranularity.PER_TRIAL  # toggle this to switch modes


# ── Configuration ───────────────────────────────────────────────────

MAX_TRIALS = 3

TEST_CASES = [
    {
        "id": 1,
        "origin": "London",
        "cities": "Lisbon, Athens, Istanbul",
        "date_range": "September 1-5, 2026",
        "interests": "beaches, nightlife",
    },
    {
        "id": 2,
        "origin": "Mumbai",
        "cities": "Goa",
        "date_range": "September 19-25, 2026",
        "interests": "beaches, nightlife",
    },
    {
        "id": 3,
        "origin": "Kolkata",
        "cities": "Mumbai , Goa",
        "date_range": "September 19-22, 2026",
        "interests": "beaches,historical places,art,cultural places,temples",
    },
    {
        "id": 4,
        "origin": "Delhi",
        "cities": "Manali, Shimla, Rishikesh",
        "date_range": "October 10-15, 2026",
        "interests": "mountains, adventure sports, trekking, nature",
    },
    {
        "id": 5,
        "origin": "Bangalore",
        "cities": "Coorg, Wayanad",
        "date_range": "November 5-7, 2026",
        "interests": "nature, coffee plantations, wildlife, relaxation",
    },
    {
        "id": 6,
        "origin": "Bangalore",
        "cities": "Coorg, Wayanad",
        "date_range": "November 5-9, 2026",
        "interests": "nature, coffee plantations, wildlife, relaxation",
    },
    {
        "id": 7,
        "origin": "Mumbai",
        "cities": "Jaipur, Udaipur, Jodhpur",
        "date_range": "December 20-27, 2026",
        "interests": "history, forts, palaces, shopping, local cuisine",
    },
    {
        "id": 8,
        "origin": "Chennai",
        "cities": "Pondicherry, Kerala backwaters",
        "date_range": "January 3-7, 2027",
        "interests": "beaches, relaxation, French heritage, food",
    },
    {
        "id": 9,
        "origin": "Pune",
        "cities": "Lonavala, Mahabaleshwar",
        "date_range": "July 12-15, 2027",
        "interests": "monsoon views, waterfalls, short trip, nature",
    },
    {
        "id": 10,
        "origin": "New York",
        "cities": "Paris, Amsterdam",
        "date_range": "December 15-22, 2026",
        "interests": "museums, art, architecture, cafes",
    },
    {
        "id": 11,
        "origin": "Singapore",
        "cities": "Bali, Bangkok",
        "date_range": "September 8-14, 2026",
        "interests": "beaches, temples, street food",
    },
]

CHECK_KEYS = [
    "budget_ok", "days_ok", "no_duplicates",
    "constraints_ok", "feasible", "no_hallucination",
]

# ── Output file names parameterized by granularity (Change 4) ──────

def _output_paths():
    """Return CSV, JSON, and traces dir based on current granularity mode."""
    mode = REFLEXION_GRANULARITY.value  # "per_trial" or "per_episode"
    return (
        f"benchmark_{mode}.csv",
        f"benchmark_{mode}.json",
        f"traces_{mode}",
    )

CSV_COLUMNS = [
    "test_case", "trial", "overall_pass", "accuracy",
    "checks_passed", "checks_applicable",
    *CHECK_KEYS,
    "failure_reasons", "episode_reflections", "execution_time_s",
]


# ── Trace helpers ───────────────────────────────────────────────────

def save_trace(
    test_case_id: int,
    trial: int,
    crew_output: str,
    eval_result: dict,
    reflection: str = "",
    episode_reflections: List[str] = None,
):
    _, _, traces_dir = _output_paths()
    os.makedirs(traces_dir, exist_ok=True)

    prefix = f"tc{test_case_id}_reflexion_trial{trial}"

    with open(os.path.join(traces_dir, f"{prefix}_output.txt"), "w") as f:
        f.write(crew_output)

    trace = {
        "test_case_id": test_case_id,
        "trial": trial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reflexion_granularity": REFLEXION_GRANULARITY.value,
        "eval_result": {k: v for k, v in eval_result.items()
                        if k not in ("judge_raw",)},
        "crew_output_length": len(crew_output),
        "reflection_preview": reflection[:1000] if reflection else "",
        "episode_reflections_count": len(episode_reflections) if episode_reflections else 0,
    }
    trace_path = os.path.join(traces_dir, f"{prefix}_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    # Trial-level reflection
    if reflection:
        with open(os.path.join(traces_dir, f"{prefix}_reflection.txt"), "w") as f:
            f.write(reflection)

    # Episode-level reflections (Change 2 — saved separately for post-hoc analysis)
    if episode_reflections:
        with open(os.path.join(traces_dir, f"{prefix}_episode_reflections.txt"), "w") as f:
            f.write("\n---\n".join(episode_reflections))

    print(f"Trace saved: {trace_path}")


def save_results_incremental(all_results: list):
    csv_file, json_file, _ = _output_paths()
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def bundle_results_zip(all_results: list) -> str:
    """Collect all outputs into a folder and return a zip file path.

    Creates:
      results_<mode>_<timestamp>/
        summary.txt          – human-readable summary
        benchmark_<mode>.csv
        benchmark_<mode>.json
        traces_<mode>/       – all trace files
    Then zips the folder and returns the zip path.
    """
    csv_file, json_file, traces_dir = _output_paths()
    mode = REFLEXION_GRANULARITY.value
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results_{mode}_{ts}"

    os.makedirs(folder_name, exist_ok=True)

    # --- Write summary.txt ---
    buf = StringIO()
    buf.write(f"Reflexion Benchmark Summary\n")
    buf.write(f"{'='*50}\n")
    buf.write(f"Mode:      {mode.upper()}\n")
    buf.write(f"Timestamp: {ts}\n")
    buf.write(f"Test Cases: {len(set(r['test_case'] for r in all_results))}\n")
    buf.write(f"Max Trials: {MAX_TRIALS}\n\n")

    by_tc = defaultdict(list)
    for r in all_results:
        by_tc[r["test_case"]].append(r)

    pass_count = sum(
        1 for trials in by_tc.values()
        if any(t["overall_pass"] for t in trials)
    )
    total_tc = len(by_tc)
    pass_rate = pass_count / total_tc if total_tc else 0

    buf.write(f"Pass@1 (eventual): {pass_count}/{total_tc} = {pass_rate:.4f}\n\n")

    buf.write(f"{'TC':<5} {'Trial':<7} {'Pass':<6} {'Accuracy':<10} "
              f"{'Time(s)':<8} {'Episodes':<12} {'Failures'}\n")
    buf.write(f"{'─'*5} {'─'*7} {'─'*6} {'─'*10} {'─'*8} {'─'*12} {'─'*30}\n")

    for r in all_results:
        passed = "Yes" if r["overall_pass"] else "NO"
        failures = r["failure_reasons"][:50] if r["failure_reasons"] else "—"
        episodes = r.get("episode_reflections", "—")
        buf.write(f"{r['test_case']:<5} {r['trial']:<7} {passed:<6} "
                  f"{r['accuracy']:<10.4f} {r['execution_time_s']:<8} "
                  f"{str(episodes):<12} {failures}\n")

    buf.write(f"\nAccuracy Progression:\n")
    for tc_id, trials in sorted(by_tc.items()):
        accs = [f"{t['accuracy']:.2f}" for t in sorted(trials, key=lambda x: x['trial'])]
        best = max(t['accuracy'] for t in trials)
        buf.write(f"  TC {tc_id}: {' → '.join(accs)}  (best: {best:.2f})\n")

    times = [r["execution_time_s"] for r in all_results]
    avg_time = sum(times) / len(times) if times else 0
    buf.write(f"\nAvg time per trial: {avg_time:.1f}s\n")
    buf.write(f"Total benchmark time: {sum(times):.1f}s\n")

    with open(os.path.join(folder_name, "summary.txt"), "w") as f:
        f.write(buf.getvalue())

    # --- Copy CSV and JSON ---
    if os.path.exists(csv_file):
        shutil.copy2(csv_file, folder_name)
    if os.path.exists(json_file):
        shutil.copy2(json_file, folder_name)

    # --- Copy traces directory ---
    if os.path.isdir(traces_dir):
        shutil.copytree(traces_dir, os.path.join(folder_name, os.path.basename(traces_dir)),
                        dirs_exist_ok=True)

    # --- Zip the folder ---
    zip_path = f"{folder_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_name))
                zf.write(file_path, arcname)

    # Clean up the unzipped folder
    shutil.rmtree(folder_name)

    print(f"\n📦 Bundled results into {zip_path}")
    return zip_path


# ── Crew runner (Change 1.4 — hierarchical hub topology) ───────────

# Module-level state shared between _run_single_crew and episode_callback.
# Reset at the start of each trial in run_reflexion().
_memory: ReflexionMemory = None
_seen_venues: set = set()


def episode_callback(task_output):
    """CrewAI task_callback — invoked after each agent step completes.

    In PER_EPISODE mode, runs a cheap rule-based eval and, if it finds
    a problem, generates an episode-level reflection that subsequent
    agents in this same trial will see.
    """
    if REFLEXION_GRANULARITY != ReflexionGranularity.PER_EPISODE:
        return

    global _seen_venues, _memory

    agent_name = str(getattr(task_output, "agent", "unknown"))
    agent_output = str(task_output)
    agent_input = str(getattr(task_output, "description", ""))

    mini_eval = evaluate_episode(agent_name, agent_output, _seen_venues)

    if not mini_eval["ok"]:
        print(f"    🔍 Episode eval flagged issues for {agent_name}: {mini_eval['duplicates_found']}")
        reflection = _memory.reflect_episode(
            agent_name=agent_name,
            agent_input=agent_input,
            agent_output=agent_output,
            mini_eval=mini_eval,
        )
        _memory.store_episode_reflection(reflection)
        print(f"    🪞 Episode reflection stored: {reflection[:200]}...")


def _build_task_description(tc: dict) -> str:
    return (
        f"Plan a trip from {tc['origin']} to one of "
        f"{tc['cities']} during {tc['date_range']}, "
        f"interests: {tc['interests']}"
    )


def _run_single_crew(test_case: dict, reflexion_context: str = "") -> str:
    """Run the 5-agent trip planner crew once in hierarchical mode.

    travel_concierge is the manager/hub; the four spoke agents
    (city_selection, local_expert, weather, messaging) are delegated to.
    """
    global _memory

    agents = TripAgents()
    tasks = TripTasks()

    # Spoke agents
    city_selector = agents.city_selection_agent()
    local_expert = agents.local_expert()
    weather = agents.weather_agent()
    messaging = agents.messaging_agent()

    # Hub / manager
    travel_concierge = agents.travel_concierge()

    # Prepend episode context (only non-empty in PER_EPISODE mode)
    episode_ctx = _memory.get_episode_context() if _memory else ""
    full_context = episode_ctx + reflexion_context

    # Tasks
    identify = tasks.identify_task(
        city_selector,
        test_case["origin"],
        test_case["cities"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=full_context,
    )
    gather = tasks.gather_task(
        local_expert,
        test_case["origin"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=full_context,
    )
    weather_task = tasks.weather_task(
        weather,
        test_case["cities"],
        test_case["date_range"],
        extra_context=full_context,
    )
    plan = tasks.plan_task(
        travel_concierge,
        test_case["origin"],
        test_case["interests"],
        test_case["date_range"],
        extra_context=full_context,
    )
    message = tasks.messaging_task(
        messaging,
        extra_context=full_context,
    )

    crew = Crew(
        agents=[city_selector, local_expert, weather, messaging],
        tasks=[identify, gather, weather_task, plan, message],
        process=Process.hierarchical,
        manager_agent=travel_concierge,
        task_callback=episode_callback,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# ── Reflexion loop ──────────────────────────────────────────────────

def run_reflexion(test_case: dict, max_trials: int = MAX_TRIALS) -> List[dict]:
    global _memory, _seen_venues

    task_input = {
        "origin": test_case["origin"],
        "cities": test_case["cities"],
        "date_range": test_case["date_range"],
        "interests": test_case["interests"],
    }
    task_desc = _build_task_description(test_case)

    memory = ReflexionMemory()
    memory.clear_session()
    _memory = memory  # expose to episode_callback
    trial_results = []

    # --- Trial 0 (initial trajectory) ---
    print(f"\n     Trial 0 (initial trajectory)...")

    _seen_venues = set()
    memory.clear_episode_reflections()

    reflexion_context = memory.build_reflexion_context(task_desc)

    start = time.time()
    try:
        output = _run_single_crew(test_case, reflexion_context=reflexion_context)
    except Exception as e:
        print(f"     Trial 0 FAILED: {e}")
        output = f"[ERROR] Crew run failed: {e}"
    elapsed = time.time() - start

    eval_result = evaluate(task_input, output)
    print(f"     Trial 0: accuracy={eval_result['accuracy']:.2f}, "
          f"pass={eval_result['overall_pass']}")

    # Collect episode reflections for this trial
    episode_refs = list(memory._episode_reflections)

    trial_results.append({
        "test_case": test_case["id"],
        "trial": 0,
        "overall_pass": eval_result["overall_pass"],
        "accuracy": eval_result["accuracy"],
        "checks_passed": eval_result["checks_passed"],
        "checks_applicable": eval_result["checks_applicable"],
        "failure_reasons": "; ".join(eval_result["failure_reasons"]),
        "episode_reflections": f"{len(episode_refs)} episode reflections",
        "execution_time_s": round(elapsed, 1),
        **{ck: eval_result.get(ck) for ck in CHECK_KEYS},
    })

    # Trial-level reflection (runs in BOTH modes)
    reflection_text = ""
    if not eval_result["overall_pass"]:
        print(f"     Generating self-reflection sr_0...")
        reflection_text = memory.reflect(task_desc, output, eval_result)
        memory.store(task_desc, output, reflection_text, eval_result)

    save_trace(test_case["id"], 0, output, eval_result,
               reflection_text, episode_refs)

    if eval_result["failure_reasons"]:
        for fr in eval_result["failure_reasons"]:
            print(f"       → {fr}")

    # --- Retry trials ---
    t = 0
    while not eval_result["overall_pass"] and t < max_trials:
        t += 1
        print(f"\n     Trial {t}/{max_trials} (reflexion retry)...")

        # Reset episode state for this trial
        _seen_venues = set()
        memory.clear_episode_reflections()

        reflexion_context = memory.build_reflexion_context(task_desc)

        start = time.time()
        try:
            output = _run_single_crew(test_case, reflexion_context=reflexion_context)
        except Exception as e:
            print(f"     Trial {t} FAILED: {e}")
            output = f"[ERROR] Crew run failed: {e}"
        elapsed = time.time() - start

        eval_result = evaluate(task_input, output)
        print(f"     Trial {t}: accuracy={eval_result['accuracy']:.2f}, "
              f"pass={eval_result['overall_pass']}")

        episode_refs = list(memory._episode_reflections)

        trial_results.append({
            "test_case": test_case["id"],
            "trial": t,
            "overall_pass": eval_result["overall_pass"],
            "accuracy": eval_result["accuracy"],
            "checks_passed": eval_result["checks_passed"],
            "checks_applicable": eval_result["checks_applicable"],
            "failure_reasons": "; ".join(eval_result["failure_reasons"]),
            "episode_reflections": f"{len(episode_refs)} episode reflections",
            "execution_time_s": round(elapsed, 1),
            **{ck: eval_result.get(ck) for ck in CHECK_KEYS},
        })

        # Trial-level reflection (runs in BOTH modes)
        reflection_text = ""
        if not eval_result["overall_pass"]:
            print(f"    🪞 Generating self-reflection sr_{t}...")
            reflection_text = memory.reflect(task_desc, output, eval_result)
            memory.store(task_desc, output, reflection_text, eval_result)
        else:
            print(f"    ✅ PASSED at trial {t}!")

        save_trace(test_case["id"], t, output, eval_result,
                   reflection_text, episode_refs)

        if eval_result["failure_reasons"]:
            for fr in eval_result["failure_reasons"]:
                print(f"       → {fr}")

    return trial_results


# ── Summary ─────────────────────────────────────────────────────────

def print_summary(all_results: list):
    """Print accuracy summary."""
    mode_label = REFLEXION_GRANULARITY.value.upper()
    print(f"\n{'='*70}")
    print(f"  REFLEXION SUMMARY ({mode_label})")
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

    print(f"  Mode: {mode_label}")
    print(f"  Pass@1 (eventual pass any trial): {pass_count}/{total_tc} = {pass_at_1:.4f}\n")

    print(f"  {'TC':<5} {'Trial':<7} {'Pass':<6} {'Accuracy':<10} "
          f"{'Time(s)':<8} {'Episodes':<10} {'Failures'}")
    print(f"  {'─'*5} {'─'*7} {'─'*6} {'─'*10} {'─'*8} {'─'*10} {'─'*30}")

    for r in all_results:
        passed = "Yes" if r["overall_pass"] else "NO"
        failures = r["failure_reasons"][:50] if r["failure_reasons"] else "—"
        episodes = r.get("episode_reflections", "—")
        print(f"  {r['test_case']:<5} {r['trial']:<7} {passed:<6} "
              f"{r['accuracy']:<10.4f} {r['execution_time_s']:<8} {episodes:<10} {failures}")

    # Accuracy progression
    print(f"\n  {'─'*60}")
    print(f"  ACCURACY PROGRESSION:\n")
    for tc_id, trials in sorted(by_tc.items()):
        accs = [f"{t['accuracy']:.2f}" for t in sorted(trials, key=lambda x: x['trial'])]
        best = max(t['accuracy'] for t in trials)
        print(f"  TC {tc_id}: {' → '.join(accs)}  (best: {best:.2f})")

    # Per-check pass rate
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

    # Timing
    print(f"\n  {'─'*60}")
    times = [r["execution_time_s"] for r in all_results]
    avg_time = sum(times) / len(times) if times else 0
    total_time = sum(times)
    print(f"  Avg time per trial:  {avg_time:.1f}s")
    print(f"  Total benchmark time: {total_time:.1f}s")
    print()


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_tc = len(TEST_CASES)
    mode_label = REFLEXION_GRANULARITY.value.upper()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  BENCHMARK: Reflexion ({mode_label})" + " " * (37 - len(mode_label)) + "║")
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
    csv_file, json_file, traces_dir = _output_paths()
    print(f"\n CSV saved to {csv_file}")
    print(f" JSON saved to {json_file}")
    print(f" Traces saved to {traces_dir}/")

    print_summary(all_results)

    # Bundle everything into a zip
    zip_path = bundle_results_zip(all_results)
    print(f"\n Results zip: {zip_path}")
