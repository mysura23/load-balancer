
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import statistics
from datetime import datetime


@dataclass
class Task:
    pid: int
    burst_time: int
    remaining_time: int
    priority: int
    arrival_step: int
    assigned_cpu: Optional[int] = None
    start_step: Optional[int] = None
    finish_step: Optional[int] = None

    @property
    def waiting_time(self) -> int:
        if self.start_step is None:
            return 0
        return self.start_step - self.arrival_step

    @property
    def turnaround_time(self) -> int:
        if self.finish_step is None:
            return 0
        return self.finish_step - self.arrival_step


@dataclass
class Processor:
    cpu_id: int
    queue: List[Task] = field(default_factory=list)
    busy_time: int = 0
    context_switches: int = 0

    def enqueue_task(self, task: Task):
        self.queue.append(task)
        task.assigned_cpu = self.cpu_id

    def dequeue_task(self) -> Optional[Task]:
        if not self.queue:
            return None
        return self.queue.pop(0)

    def current_load(self) -> int:
        return sum(t.remaining_time for t in self.queue)


class LoadBalancer:
    """Core engine for dynamic load balancing simulation."""

    def __init__(self, cpu_count: int = 4, strategy: str = "least_loaded"):
        self.time_step: int = 0
        self.strategy: str = strategy
        self.processors: List[Processor] = [Processor(i) for i in range(cpu_count)]
        self.next_pid: int = 1
        self.completed_tasks: List[Task] = []
        self.log_messages: List[str] = []
        self._rr_index: int = 0

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{ts}] {msg}")

    def create_random_task(self, burst_min: int = 2, burst_max: int = 15, priority_max: int = 5) -> Task:
        burst = random.randint(burst_min, burst_max)
        prio = random.randint(1, priority_max)
        t = Task(
            pid=self.next_pid,
            burst_time=burst,
            remaining_time=burst,
            priority=prio,
            arrival_step=self.time_step,
        )
        self.next_pid += 1
        self.assign_task(t)
        self._log(f"Created Task P{t.pid} (burst={burst}, prio={prio})")
        return t

    def create_task(self, burst: int, priority: int) -> Task:
        t = Task(
            pid=self.next_pid,
            burst_time=burst,
            remaining_time=burst,
            priority=priority,
            arrival_step=self.time_step,
        )
        self.next_pid += 1
        self.assign_task(t)
        self._log(f"Created Task P{t.pid} (burst={burst}, prio={priority})")
        return t

    def assign_task(self, task: Task):
        if self.strategy == "least_loaded":
            cpu = min(self.processors, key=lambda c: c.current_load())
        elif self.strategy == "round_robin":
            cpu = self.processors[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self.processors)
        elif self.strategy == "random":
            cpu = random.choice(self.processors)
        elif self.strategy == "shortest_queue":
            cpu = min(self.processors, key=lambda c: len(c.queue))
        else:
            cpu = min(self.processors, key=lambda c: c.current_load())
        cpu.enqueue_task(task)

    def step(self, time_quantum: int = 1, enable_stealing: bool = True):
        self.time_step += 1
        for cpu in self.processors:
            if not cpu.queue:
                continue
            current = cpu.queue[0]
            if current.start_step is None:
                current.start_step = self.time_step

            work = min(time_quantum, current.remaining_time)
            current.remaining_time -= work
            cpu.busy_time += work

            if current.remaining_time <= 0:
                current.finish_step = self.time_step
                self.completed_tasks.append(current)
                cpu.dequeue_task()
                cpu.context_switches += 1
                self._log(f"Task P{current.pid} finished on CPU{cpu.cpu_id}.")

        if enable_stealing:
            self._work_stealing()

    def _work_stealing(self):
        loads = [cpu.current_load() for cpu in self.processors]
        if not any(loads):
            return

        max_cpu = max(self.processors, key=lambda c: c.current_load())
        min_cpu = min(self.processors, key=lambda c: c.current_load())
        if max_cpu.current_load() - min_cpu.current_load() < 2:
            return

        if len(max_cpu.queue) > 1:
            stolen = max_cpu.queue.pop()
            min_cpu.enqueue_task(stolen)
            self._log(
                f"Work stealing: moved P{stolen.pid} from CPU{max_cpu.cpu_id} "
                f"to CPU{min_cpu.cpu_id}."
            )

    def loads(self):
        return [c.current_load() for c in self.processors]

    def utilization(self):
        if self.time_step == 0:
            return [0.0 for _ in self.processors]
        return [cpu.busy_time / self.time_step for cpu in self.processors]

    def imbalance_factor(self) -> float:
        lds = self.loads()
        if not any(lds):
            return 0.0
        return statistics.pstdev(lds)

    def summary(self) -> Dict[str, float]:
        if not self.completed_tasks:
            return {"completed": 0, "avg_waiting": 0.0, "avg_turnaround": 0.0}
        waits = [t.waiting_time for t in self.completed_tasks]
        turns = [t.turnaround_time for t in self.completed_tasks]
        return {
            "completed": len(self.completed_tasks),
            "avg_waiting": sum(waits) / len(waits),
            "avg_turnaround": sum(turns) / len(turns),
        }

    def reset(self, cpu_count: Optional[int] = None):
        if cpu_count is None:
            cpu_count = len(self.processors)
        self.__init__(cpu_count=cpu_count, strategy=self.strategy)

    def clear_queues(self):
        for cpu in self.processors:
            cpu.queue = []
        self._log("Cleared all CPU queues.")
