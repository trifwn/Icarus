# --- Example Implementation: Fibonacci Calculation ---
from ICARUS.computation.task import Task, TaskExecutor, ProgressUpdate, SimulationRunner, ExecutionMode
import os
import json
import time
import tempfile

# --- Example Implementation: Summation Calculation ---


def register_progress(work_dir: str, task_id: int, step: int, completed: bool = False, error: str | None = None):
    """Writes the current progress of a task to a dedicated JSON file."""
    # Ensure the working directory exists
    os.makedirs(work_dir, exist_ok=True)
    progress_file = os.path.join(work_dir, f"task_{task_id}_progress.json")
    with open(progress_file, "w") as f:
        json.dump({"current_step": step, "completed": completed, "error": error}, f)


def update_progress(task: Task) -> ProgressUpdate:
    """
    Probe function: Reads the progress file for a task and returns a ProgressUpdate.
    This function is called by the ProgressMonitor.
    """
    # Unpack args safely
    n, work_dir, _ = task.args

    progress_file = os.path.join(work_dir, f"task_{task.id_num}_progress.json")
    if not os.path.exists(progress_file):
        return ProgressUpdate(task_id=task.id_num, current_iteration=0, max_iterations=n, name=task.name)
    try:
        with open(progress_file, "r") as f:
            data = json.load(f)
        return ProgressUpdate(
            task_id=task.id_num,
            current_iteration=data.get("current_step", 0),
            max_iterations=n,
            name=task.name,
            error=data.get("error"),
            completed=data.get("completed", False),
        )
    except (IOError, json.JSONDecodeError) as e:
        return ProgressUpdate(
            task_id=task.id_num,
            current_iteration=0,
            max_iterations=n,
            name=task.name,
            error=str(e),
        )


def calculate_sum_and_report_progress(n: int, workdir: str, task_id: int) -> int:
    """
    Calculates the sum of integers from 0 to n-1 and reports progress.
    """
    total_sum = 0
    # Initial progress registration
    register_progress(workdir, task_id, 0, completed=False)
    for i in range(n):
        try:
            total_sum += i
            # Simulate some work
            time.sleep(0.01)
            # Report progress after each step
            register_progress(workdir, task_id, i + 1, completed=(i + 1 == n))
        except Exception as e:
            register_progress(workdir, task_id, i + 1, completed=True, error=str(e))
            return -1
    return total_sum


def demo_runner():
    """Demonstrates using the SimulationRunner with the calculation task."""
    targets = [250, 300, 280, 320, 270, 350, 290, 310]

    # Use a temporary directory that persists for the duration of the script run
    temp_dir = os.getcwd()
    # with tempfile.TemporaryDirectory() as temp_dir:
    work_dir = os.path.join(temp_dir, "demo_tasks")
    os.makedirs(work_dir, exist_ok=True)

    # 1. Create a list of tasks
    tasks = []
    for i, n in enumerate(targets):
        task_name = f"Sum({n})"
        # We MUST pass the task's own ID (i) so it can create a unique progress file.
        task_args = (n, work_dir, i)
        task = Task(
            name=task_name,
            id_num=i,
            progress_probe=update_progress,
            execution_function=calculate_sum_and_report_progress,
            args=task_args,
        )
        tasks.append(task)

    # 2. Create a specific executor instance (optional, but good practice)
    # The executor uses the same 'update_progress' probe to report final status
    task_executor = TaskExecutor(update_progress, delay_per_step=0.02)

    # 3. Configure and run the simulation for all execution modes
    for execution_mode in [ExecutionMode.SEQUENTIAL, ExecutionMode.THREADING, ExecutionMode.MULTIPROCESSING]:
        print(f"\n--- DEMONSTRATING {execution_mode.name} EXECUTION ---")
        runner = SimulationRunner(execution_mode=execution_mode)
        runner.set_executor(task_executor).add_tasks(tasks)
        results = runner.run()
        runner.print_summary()

        # Optionally, inspect individual results
        for result in results:
            print(f"Result for task '{result.name}': {result.result_value}")


if __name__ == "__main__":
    demo_runner()
