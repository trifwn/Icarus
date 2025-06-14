# class ProgressMonitor:
#     """
#     Centralized progress monitoring using tqdm.
#     It runs in a separate thread and polls tasks for updates via their 'probe'.
#     """

#     def __init__(self, tasks: list[Task], refresh_rate: float = 0.5):
#         self.tasks = tasks
#         self.refresh_rate = refresh_rate
#         self.progress_bars: dict[int, tqdm] = {}
#         self.stop_event = Event()
#         self.logger = logging.getLogger(__name__)

#     def __enter__(self):
#         """Context manager entry - create progress bars."""
#         for i, task in enumerate(self.tasks):
#             # Use a dynamic position for each bar
#             pbar = tqdm(
#                 total=100,  # We'll use percentage as the total
#                 desc=f"{task.name}",
#                 position=i,
#                 leave=True,
#                 colour="#cc3300",
#                 bar_format="{l_bar}{bar:30}{r_bar}",
#             )
#             self.progress_bars[task.id_num] = pbar
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Context manager exit - cleanup progress bars."""
#         self.stop_event.set()
#         # A brief pause to allow final updates to be rendered
#         sleep(0.1)
#         for pbar in self.progress_bars.values():
#             if pbar.n < pbar.total and not pbar.disable:
#                 pbar.n = pbar.total
#                 pbar.refresh()
#             pbar.close()

#     def _update_pbar(self, update: ProgressUpdate):
#         """Applies a ProgressUpdate to its corresponding tqdm bar."""
#         if update.task_id not in self.progress_bars:
#             return

#         pbar = self.progress_bars[update.task_id]
#         percentage = update.progress_percentage if update.progress_percentage is not None else 0

#         if update.error:
#             pbar.set_description(f"{update.name} - ERROR")
#             pbar.colour = "#ff0000"  # Red
#             pbar.n = int(percentage)
#         elif update.completed:
#             pbar.n = 100
#             pbar.set_description(f"{update.name} - DONE")
#             pbar.colour = "#00ff00"  # Green
#         else:
#             pbar.n = int(percentage)
#             progress_text = f"{update.name} - {update.current_iteration}/{update.max_iterations}"
#             pbar.set_description(progress_text)

#         pbar.refresh()

#     def monitor_loop(self):
#         """Main monitoring loop, polls each task's probe."""
#         while not self.stop_event.is_set():
#             all_completed = True
#             for task in self.tasks:
#                 if self.stop_event.is_set():
#                     break
#                 try:
#                     update = task.progress_probe(task)
#                     self._update_pbar(update)
#                     if not update.completed:
#                         all_completed = False
#                 except Exception as e:
#                     self.logger.debug(f"Error probing task {task.id_num}: {e}")

#             if all_completed:
#                 sleep(self.refresh_rate)  # one last sleep to show 100%
#                 break
#             sleep(self.refresh_rate)


# class SimulationRunner:
#     """
#     The main orchestrator for running, monitoring, and managing tasks.
#     """

#     def __init__(
#         self,
#         max_workers: Optional[int] = None,
#         execution_mode: ExecutionMode = ExecutionMode.MULTIPROCESSING,
#         progress_refresh_rate: float = 0.2,
#         enable_progress_monitoring: bool = True,
#     ):
#         self.max_workers = max_workers or min(os.cpu_count() or 4, 8)
#         self.execution_mode = execution_mode
#         self.progress_refresh_rate = progress_refresh_rate
#         self.enable_progress_monitoring = enable_progress_monitoring
#         self.stop_event = Event()
#         self.tasks: List[Task] = []
#         self.results: List[TaskResult] = []
#         self.executor: Optional[TaskExecutor] = None
#         self.monitor_thread: Optional[Thread] = None
#         self.logger = logging.getLogger(__name__)
#         if not self.logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#             handler.setFormatter(formatter)
#             self.logger.addHandler(handler)
#             self.logger.setLevel(logging.INFO)

#     def _setup_signal_handlers(self):
#         """Setup signal handlers for a clean shutdown on CTRL+C."""

#         def signal_handler(signum, frame):
#             self.logger.warning("\nShutdown signal received. Cleaning up...")
#             self.stop_event.set()

#         signal.signal(signal.SIGINT, signal_handler)
#         signal.signal(signal.SIGTERM, signal_handler)

#     def add_task(self, task: Task) -> "SimulationRunner":
#         """Add a single task to the runner (builder pattern)."""
#         self.tasks.append(task)
#         return self

#     def add_tasks(self, tasks: List[Task]) -> "SimulationRunner":
#         """Add multiple tasks to the runner (builder pattern)."""
#         self.tasks.extend(tasks)
#         return self

#     def set_executor(self, executor: TaskExecutor) -> "SimulationRunner":
#         """Set a custom task executor for all tasks (builder pattern)."""
#         self.executor = executor
#         return self

#     def _execute_multiprocessing(self) -> List[TaskResult]:
#         """Execute tasks using a multiprocessing pool."""
#         self.logger.info(f"Starting multiprocessing with {self.max_workers} workers.")
#         tasks_with_executor = [(task, self.executor) for task in self.tasks]
#         with Pool(processes=self.max_workers) as pool:
#             return pool.map(worker_task_executor, tasks_with_executor)

#     def _execute_threading(self) -> List[TaskResult]:
#         """Execute tasks using a thread pool."""
#         self.logger.info(f"Starting threading with {self.max_workers} workers.")
#         results = []
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             tasks_with_executor = [(task, self.executor) for task in self.tasks]
#             futures = [executor.submit(worker_task_executor, t) for t in tasks_with_executor]
#             for future in futures:
#                 if self.stop_event.is_set():
#                     break
#                 try:
#                     results.append(future.result())
#                 except Exception as e:
#                     self.logger.error(f"A task failed in the thread pool: {e}")
#         return results

#     def _execute_sequential(self) -> List[TaskResult]:
#         """Execute tasks one by one in the main thread."""
#         self.logger.info("Starting sequential execution.")
#         results = []
#         for task in self.tasks:
#             if self.stop_event.is_set():
#                 break
#             results.append(worker_task_executor((task, self.executor)))
#         return results

#     def _start_progress_monitoring(self):
#         """Start the progress monitor in a separate thread if enabled."""
#         if not self.enable_progress_monitoring or not self.tasks:
#             return
#         self.progress_monitor = ProgressMonitor(self.tasks, self.progress_refresh_rate)

#         def monitor_runner():
#             with self.progress_monitor:
#                 self.progress_monitor.monitor_loop()

#         self.monitor_thread = Thread(target=monitor_runner, daemon=True)
#         self.monitor_thread.start()

#     def run(self) -> List[TaskResult]:
#         """Execute all added tasks using the configured mode and return the results."""
#         if not self.tasks:
#             self.logger.warning("No tasks to execute.")
#             return []
#         self.logger.info(f"Starting execution of {len(self.tasks)} tasks in '{self.execution_mode.value}' mode.")
#         self._setup_signal_handlers()
#         self._start_progress_monitoring()
#         execution_map = {
#             ExecutionMode.MULTIPROCESSING: self._execute_multiprocessing,
#             ExecutionMode.THREADING: self._execute_threading,
#             ExecutionMode.SEQUENTIAL: self._execute_sequential,
#         }
#         try:
#             execute_func = execution_map.get(self.execution_mode)
#             if not execute_func:
#                 raise ValueError(f"Unknown execution mode: {self.execution_mode}")
#             self.results = execute_func()
#         except Exception as e:
#             self.logger.error(f"A critical error occurred during execution: {e}")
#         finally:
#             self.stop_event.set()
#             if self.monitor_thread and self.monitor_thread.is_alive():
#                 self.monitor_thread.join(timeout=1.0)
#             self.logger.info("Execution finished.")
#         return self.results

#     def get_summary(self) -> Dict[str, Any]:
#         """Get execution summary statistics."""
#         if not self.results:
#             return {"total_tasks": len(self.tasks), "successful": 0, "failed": 0}
#         successful = [r for r in self.results if r.success]
#         total_time = sum(r.execution_time for r in self.results if r.execution_time)
#         return {
#             "total_tasks": len(self.results),
#             "successful": len(successful),
#             "failed": len(self.results) - len(successful),
#             "success_rate": len(successful) / len(self.results) * 100 if self.results else 0,
#             "total_execution_time": total_time,
#             "average_execution_time": total_time / len(self.results) if self.results else 0,
#             "execution_mode": self.execution_mode.value,
#         }

#     def print_summary(self):
#         """Print a formatted execution summary to the console."""
#         summary = self.get_summary()
#         print("\n" + "=" * 60 + "\nEXECUTION SUMMARY\n" + "=" * 60)
#         for key, value in summary.items():
#             key_str = key.replace("_", " ").title()
#             if isinstance(value, float):
#                 print(f"{key_str:<25}: {value:.2f}")
#             else:
#                 print(f"{key_str:<25}: {value}")
#         print("-" * 60)
