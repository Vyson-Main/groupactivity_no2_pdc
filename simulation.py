import time
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

NUM_SAMPLES = 200          # Total blood samples to analyze
NUM_WORKERS = 4            # Number of parallel lab technicians (threads)
ANALYSIS_TIME = 0.01       # Simulated time (seconds) per sample analysis step

results = {}
results_lock = threading.Lock()


def analyze_sample(sample_id: int) -> dict:
    """
    Simulates the analysis of a single blood sample.
    Each sample goes through: centrifuge → measure → classify.
    This is the smallest independent unit of computation.
    """
    time.sleep(ANALYSIS_TIME)
    rbc = round(random.uniform(4.0, 6.0), 2)   # Red blood cells (M/uL)

    time.sleep(ANALYSIS_TIME)
    wbc = round(random.uniform(4.5, 11.0), 2)  # White blood cells (K/uL)

    time.sleep(ANALYSIS_TIME)
    status = "Normal" if 4.5 <= wbc <= 11.0 and 4.0 <= rbc <= 6.0 else "Abnormal"

    return {"sample_id": sample_id, "RBC": rbc, "WBC": wbc, "status": status}

def run_sequential(num_samples: int) -> float:
    """One technician processes all samples one at a time."""
    print(f"\n[SEQUENTIAL] Processing {num_samples} samples with 1 worker...")
    seq_results = {}
    start = time.perf_counter()

    for sid in range(num_samples):
        seq_results[sid] = analyze_sample(sid)

    elapsed = time.perf_counter() - start
    print(f"[SEQUENTIAL] Done. Time: {elapsed:.4f}s | Samples processed: {len(seq_results)}")
    return elapsed

def worker_task(sample_ids: list) -> list:
    """Each worker (thread) processes its assigned partition of samples."""
    local_results = []
    for sid in sample_ids:
        result = analyze_sample(sid)
        local_results.append(result)
    return local_results


def run_parallel(num_samples: int, num_workers: int) -> float:
    """Multiple technicians each process a partition of samples concurrently."""
    print(f"\n[PARALLEL]   Processing {num_samples} samples with {num_workers} workers...")

    partitions = [[] for _ in range(num_workers)]
    for sid in range(num_samples):
        partitions[sid % num_workers].append(sid)

    par_results = {}
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_task, part): i for i, part in enumerate(partitions)}
        for future in as_completed(futures):
            worker_results = future.result()
            with results_lock:
                for r in worker_results:
                    par_results[r["sample_id"]] = r

    elapsed = time.perf_counter() - start
    print(f"[PARALLEL]   Done. Time: {elapsed:.4f}s | Samples processed: {len(par_results)}")
    return elapsed

def benchmark():
    print("=" * 60)
    print("   HOSPITAL LAB SAMPLE ANALYSIS — BENCHMARK REPORT")
    print("=" * 60)
    print(f"   Total Samples : {NUM_SAMPLES}")
    print(f"   Workers (Parallel): {NUM_WORKERS}")
    print(f"   Time per analysis step: {ANALYSIS_TIME}s × 3 steps/sample")
    print("=" * 60)

    seq_time = run_sequential(NUM_SAMPLES)
    par_time = run_parallel(NUM_SAMPLES, NUM_WORKERS)

    speedup = seq_time / par_time
    ideal_speedup = NUM_WORKERS
    efficiency = (speedup / ideal_speedup) * 100

    print("\n" + "=" * 60)
    print("   RESULTS")
    print("=" * 60)
    print(f"   Sequential Time : {seq_time:.4f} seconds")
    print(f"   Parallel Time   : {par_time:.4f} seconds")
    print(f"   Speedup Ratio   : {speedup:.2f}x")
    print(f"   Ideal Speedup   : {ideal_speedup:.2f}x  ({NUM_WORKERS} workers)")
    print(f"   Efficiency      : {efficiency:.1f}%")
    print("=" * 60)

    print("\n[ANALYSIS]")
    if speedup >= ideal_speedup * 0.8:
        print(f"  Near-ideal speedup achieved ({speedup:.2f}x vs ideal {ideal_speedup}x).")
        print("  The workload is highly parallelizable with minimal synchronization overhead.")
    else:
        print(f"  Speedup ({speedup:.2f}x) is below ideal ({ideal_speedup}x).")
        print("  Overhead sources: thread spawning, lock contention on shared results dict,")
        print("  and Python GIL limitations on CPU-bound simulation steps.")
        print("  In a real I/O-bound scenario (actual lab equipment), speedup would be higher.")

    print("\n[REAL-WORLD INSIGHT]")
    print("  In a hospital lab, the bottleneck is a single technician processing samples")
    print("  serially. Assigning each technician a partition (data parallelism) allows")
    print("  simultaneous processing, dramatically reducing patient wait times.")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()
