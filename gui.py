
from __future__ import annotations
import sys
import random
from typing import List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QComboBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QFrame, QCheckBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from load_balancer_core import LoadBalancer


class BarChartCanvas(FigureCanvas):
    def __init__(self, title: str, ylabel: str, parent=None):
        self.fig = Figure(figsize=(4, 3), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.title = title
        self.ylabel = ylabel
        super().__init__(self.fig)
        self.setParent(parent)

    def update_bars(self, values: List[float], labels: List[str]):
        self.ax.clear()
        self.ax.bar(labels, values)
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_ylim(bottom=0)
        self.ax.grid(True, axis="y")
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Load Balancing in Multiprocessor Systems - Simulator")
        self.setMinimumSize(1200, 720)

        self.balancer = LoadBalancer(cpu_count=4, strategy="least_loaded")

        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self.auto_step)

        root = QWidget()
        main_layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar_layout = QVBoxLayout(sidebar)

        title = QLabel("Load Balancer Controls")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title)

        cpu_row = QHBoxLayout()
        cpu_row.addWidget(QLabel("CPUs:"))
        self.spin_cpus = QSpinBox()
        self.spin_cpus.setRange(1, 16)
        self.spin_cpus.setValue(4)
        cpu_row.addWidget(self.spin_cpus)
        sidebar_layout.addLayout(cpu_row)

        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Strategy:"))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["least_loaded", "round_robin", "random", "shortest_queue"])
        strat_row.addWidget(self.combo_strategy)
        sidebar_layout.addLayout(strat_row)

        self.chk_steal = QCheckBox("Enable Work Stealing")
        self.chk_steal.setChecked(True)
        sidebar_layout.addWidget(self.chk_steal)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Random tasks to add:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 50)
        self.spin_batch.setValue(5)
        batch_row.addWidget(self.spin_batch)
        sidebar_layout.addLayout(batch_row)

        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Burst:"))
        self.spin_burst = QSpinBox()
        self.spin_burst.setRange(1, 100)
        self.spin_burst.setValue(10)
        custom_row.addWidget(self.spin_burst)
        custom_row.addWidget(QLabel("Priority:"))
        self.spin_priority = QSpinBox()
        self.spin_priority.setRange(1, 10)
        self.spin_priority.setValue(3)
        custom_row.addWidget(self.spin_priority)
        sidebar_layout.addLayout(custom_row)

        quantum_row = QHBoxLayout()
        quantum_row.addWidget(QLabel("Time quantum:"))
        self.spin_quantum = QSpinBox()
        self.spin_quantum.setRange(1, 10)
        self.spin_quantum.setValue(1)
        quantum_row.addWidget(self.spin_quantum)
        sidebar_layout.addLayout(quantum_row)

        self.btn_add_tasks = QPushButton("Add Random Tasks")
        self.btn_add_task = QPushButton("Add Task")
        self.btn_step = QPushButton("Step Scheduler")
        self.btn_autorun = QPushButton("Start Auto Run")
        self.btn_reset = QPushButton("Reset Simulation")
        self.btn_export = QPushButton("Export Completed CSV")
        self.btn_clear_logs = QPushButton("Clear Logs")
        self.btn_help = QPushButton("Help")
        self.btn_flush = QPushButton("Flush Queues")

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Random seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(0)
        seed_row.addWidget(self.spin_seed)
        self.btn_set_seed = QPushButton("Set Seed")
        seed_row.addWidget(self.btn_set_seed)
        sidebar_layout.addLayout(seed_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Auto-run interval (ms):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(100, 2000)
        self.spin_interval.setValue(400)
        interval_row.addWidget(self.spin_interval)
        sidebar_layout.addLayout(interval_row)

        sidebar_layout.addWidget(self.btn_add_tasks)
        sidebar_layout.addWidget(self.btn_add_task)
        sidebar_layout.addWidget(self.btn_step)
        sidebar_layout.addWidget(self.btn_autorun)
        sidebar_layout.addWidget(self.btn_reset)
        sidebar_layout.addWidget(self.btn_export)
        sidebar_layout.addWidget(self.btn_clear_logs)
        sidebar_layout.addWidget(self.btn_help)
        sidebar_layout.addWidget(self.btn_flush)

        sidebar_layout.addStretch()

        self.lbl_step = QLabel("Time Step: 0")
        self.lbl_completed = QLabel("Completed: 0")
        self.lbl_wait = QLabel("Avg Waiting: 0.0")
        self.lbl_turn = QLabel("Avg Turnaround: 0.0")
        self.lbl_imbalance = QLabel("Imbalance: 0.0")

        for lbl in (self.lbl_step, self.lbl_completed, self.lbl_wait, self.lbl_turn, self.lbl_imbalance):
            lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
            sidebar_layout.addWidget(lbl)

        center = QWidget()
        center_layout = QVBoxLayout(center)

        charts_layout = QHBoxLayout()
        self.chart_loads = BarChartCanvas("CPU Load (Remaining Time)", "Load units", self)
        self.chart_util = BarChartCanvas("CPU Utilization", "Utilization", self)
        charts_layout.addWidget(self.chart_loads)
        charts_layout.addWidget(self.chart_util)
        center_layout.addLayout(charts_layout)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["CPU", "Queue Length", "Current Task", "Context Switches", "Queue Summary"])
        center_layout.addWidget(self.table)

        lbl_logs = QLabel("Event Log")
        lbl_logs.setFont(QFont("Segoe UI", 12, QFont.Bold))
        center_layout.addWidget(lbl_logs)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        center_layout.addWidget(self.log_view)

        main_layout.addWidget(sidebar, 2)
        main_layout.addWidget(center, 5)

        self.btn_add_tasks.clicked.connect(self.add_random_tasks)
        self.btn_add_task.clicked.connect(self.add_custom_task)
        self.btn_step.clicked.connect(self.manual_step)
        self.btn_autorun.clicked.connect(self.toggle_autorun)
        self.btn_reset.clicked.connect(self.reset_simulation)
        self.combo_strategy.currentTextChanged.connect(self.change_strategy)
        self.spin_interval.valueChanged.connect(self.change_interval)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_clear_logs.clicked.connect(self.clear_logs)
        self.btn_help.clicked.connect(self.show_help)
        self.btn_flush.clicked.connect(self.flush_queues)
        self.btn_set_seed.clicked.connect(self.set_seed)

        self.apply_style()
        self.refresh_all()

    def apply_style(self):
        style = (
            "QMainWindow {background-color: #020617; color: #e5e7eb;}"
            "QLabel {color: #e5e7eb;}"
            "QFrame {background-color: #020617;}"
            "QPushButton {background-color: #1d4ed8; color: white; border-radius: 6px; padding: 6px 10px;}"
            "QPushButton:hover {background-color: #2563eb;}"
            "QSpinBox, QComboBox, QCheckBox {background-color: #020617; color: #e5e7eb; border: 1px solid #374151; border-radius: 4px; padding: 2px 4px;}"
            "QTableWidget {background-color: #020617; color: #e5e7eb; gridline-color: #1f2937;}"
            "QTextEdit {background-color: #020617; color: #e5e7eb; border: 1px solid #374151;}"
        )
        self.setStyleSheet(style)

    def change_strategy(self, text: str):
        self.balancer.strategy = text

    def add_random_tasks(self):
        n = self.spin_batch.value()
        for _ in range(n):
            self.balancer.create_random_task()
        self.refresh_all()

    def add_custom_task(self):
        burst = self.spin_burst.value()
        prio = self.spin_priority.value()
        self.balancer.create_task(burst, prio)
        self.refresh_all()

    def manual_step(self):
        self.balancer.step(time_quantum=self.spin_quantum.value(), enable_stealing=self.chk_steal.isChecked())
        self.refresh_all()

    def auto_step(self):
        if random.random() < 0.7:
            self.balancer.create_random_task()
        self.balancer.step(time_quantum=self.spin_quantum.value(), enable_stealing=self.chk_steal.isChecked())
        self.refresh_all()

    def toggle_autorun(self):
        if self.auto_timer.isActive():
            self.auto_timer.stop()
            self.btn_autorun.setText("Start Auto Run")
        else:
            self.auto_timer.start(self.spin_interval.value())
            self.btn_autorun.setText("Stop Auto Run")

    def reset_simulation(self):
        cpu_count = self.spin_cpus.value()
        self.balancer.reset(cpu_count=cpu_count)
        self.refresh_all()

    def change_interval(self, value: int):
        if self.auto_timer.isActive():
            self.auto_timer.setInterval(value)

    def refresh_all(self):
        self.update_metrics()
        self.update_table()
        self.update_charts()
        self.update_logs()

    def update_metrics(self):
        self.lbl_step.setText(f"Time Step: {self.balancer.time_step}")
        summary = self.balancer.summary()
        self.lbl_completed.setText(f"Completed: {summary['completed']}")
        self.lbl_wait.setText(f"Avg Waiting: {summary['avg_waiting']:.2f}")
        self.lbl_turn.setText(f"Avg Turnaround: {summary['avg_turnaround']:.2f}")
        self.lbl_imbalance.setText(f"Imbalance: {self.balancer.imbalance_factor():.2f}")

    def update_table(self):
        cpus = self.balancer.processors
        self.table.setRowCount(len(cpus))
        for row, cpu in enumerate(cpus):
            self.table.setItem(row, 0, QTableWidgetItem(f"CPU {cpu.cpu_id}"))
            self.table.setItem(row, 1, QTableWidgetItem(str(len(cpu.queue))))
            cur = cpu.queue[0].pid if cpu.queue else "-"
            self.table.setItem(row, 2, QTableWidgetItem(str(cur)))
            self.table.setItem(row, 3, QTableWidgetItem(str(cpu.context_switches)))
            summary = ", ".join([f"P{t.pid}({t.remaining_time})" for t in cpu.queue[:3]])
            self.table.setItem(row, 4, QTableWidgetItem(summary))
        self.table.resizeColumnsToContents()

    def update_charts(self):
        loads = self.balancer.loads()
        utils = self.balancer.utilization()
        labels = [f"C{cpu.cpu_id}" for cpu in self.balancer.processors]
        self.chart_loads.update_bars(loads, labels)
        self.chart_util.update_bars(utils, labels)

    def update_logs(self):
        self.log_view.setPlainText("\n".join(self.balancer.log_messages))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "completed_tasks.csv", "CSV Files (*.csv)")
        if not path:
            return
        import csv
        fields = [
            "pid","burst_time","priority","arrival_step","start_step",
            "finish_step","waiting_time","turnaround_time","assigned_cpu"
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in self.balancer.completed_tasks:
                writer.writerow({
                    "pid": t.pid,
                    "burst_time": t.burst_time,
                    "priority": t.priority,
                    "arrival_step": t.arrival_step,
                    "start_step": t.start_step or 0,
                    "finish_step": t.finish_step or 0,
                    "waiting_time": t.waiting_time,
                    "turnaround_time": t.turnaround_time,
                    "assigned_cpu": t.assigned_cpu if t.assigned_cpu is not None else -1,
                })
        QMessageBox.information(self, "Export", "CSV exported successfully.")

    def clear_logs(self):
        self.balancer.log_messages = []
        self.update_logs()

    def flush_queues(self):
        self.balancer.clear_queues()
        self.refresh_all()

    def set_seed(self):
        import random as _r
        _r.seed(self.spin_seed.value())
        self.balancer._log("Random seed set.")

    def show_help(self):
        text = (
            "Dynamic Load Balancing Simulator\n\n"
            "Controls:\n"
            "- CPUs: set number of processors and Reset.\n"
            "- Strategy: least_loaded, round_robin, random.\n"
            "- Work Stealing: move tasks from overloaded to idle CPUs.\n"
            "- Add Task: create a task with burst and priority.\n"
            "- Add Random Tasks: batch of random tasks.\n"
            "- Step Scheduler: advance one time step.\n"
            "- Auto-run: run continuously; adjust interval.\n"
            "- Export Completed CSV: save finished tasks.\n"
            "- Clear Logs: clear event log.\n\n"
            "Metrics:\n"
            "- CPU Load: sum of remaining time in each queue.\n"
            "- Utilization: busy time / total time.\n"
            "- Imbalance: standard deviation of loads."
        )
        QMessageBox.information(self, "Help", text)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
