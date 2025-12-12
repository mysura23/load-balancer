# monitor_gui_pro_accurate.py
# PRO (Ultra-Accurate) monitor — WMIC/tasklist based, improved accuracy + UI smoothing
# Based on user's original file (reference: :contentReference[oaicite:1]{index=1})
#
# Key changes:
# - Robust WMIC wrapper with timeout + retries
# - tasklist read via CSV output for stable parsing
# - Per-process CPU: multi-sample averaging + normalize by logical CPUs
# - Per-process CPU EMA smoothing to reduce spikes
# - Graph smoothing (moving-average) + reliable redraw
# - Executable path fallback using PowerShell (Get-CimInstance)
# - taskkill fallback with PowerShell Stop-Process
# - Improved UI responsiveness and stable selection preservation
#
# Works only on Windows.

import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import sys
import time
import csv
import os
import shlex
from collections import deque, defaultdict

# Config
REFRESH_MS = 1500        # overall UI refresh interval (ms)
SAMPLE_DELAY_MS = 500    # delay between WMIC samples (ms)
HISTORY_LENGTH = 60
WMIC_TIMEOUT = 4         # seconds for WMIC/subprocess calls
WMIC_RETRIES = 2         # retries on failure
CPU_EMA_ALPHA = 0.35     # smoothing factor for per-process cpu EMA
GRAPH_SMOOTH_WINDOW = 3  # moving average window for drawn graph

# ---------------- reliable subprocess helper ----------------
def run_cmd_raw(cmd, timeout=WMIC_TIMEOUT):
    """
    Run command string via subprocess; returns stdout (decoded) or empty string.
    Uses retries and timeout protection.
    """
    # If cmd is a list, use it directly; if string, pass to shell to keep compatibility
    for attempt in range(WMIC_RETRIES + 1):
        try:
            # Use shell=True for complex Windows commands (wmic, powershell strings)
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            out = proc.stdout.decode(errors="ignore")
            return out
        except subprocess.TimeoutExpired:
            # kill and retry
            continue
        except Exception:
            # other exception: retry a couple times
            continue
    return ""

# Light wrapper (keeps previous name)
def run_cmd(cmd):
    return run_cmd_raw(cmd)

# ---------------- improved process list parsing ----------------
def get_process_list_raw():
    """
    Use 'tasklist /FO CSV /NH' which outputs stable CSV,
    so we avoid brittle fixed-column parsing.
    Returns list of dicts with keys: image, pid, session_name, session_num, mem_mb, mem_str, cpu_pct, disk, net
    """
    out = run_cmd('tasklist /FO CSV /NH')
    if not out:
        return []
    parsed = []
    # CSV output lines like: "Image Name","PID","Session Name","Session#","Mem Usage"
    try:
        reader = csv.reader(out.splitlines())
        for row in reader:
            if not row:
                continue
            # Ensure row has at least 5 columns
            # Some localized Windows might change column names but order is standard
            image = row[0].strip('" ')
            pid = row[1].strip()
            session_name = row[2].strip('" ')
            session_num = row[3].strip()
            mem_raw = row[4].strip('" ').replace(" K", "K").replace("k", "K")
            mb_val, mb_str = parse_mem_to_mb(mem_raw)
            parsed.append({
                "image": image,
                "pid": pid,
                "session_name": session_name,
                "session_num": session_num,
                "mem_raw": mem_raw,
                "mem_mb": mb_val,
                "mem_str": mb_str,
                "cpu_pct": 0.0,
                "disk": "0 MB/s",
                "net": "0 Mbps"
            })
    except Exception:
        # Fallback: original parsing if CSV fails
        out2 = run_cmd('tasklist')
        lines = [ln for ln in out2.splitlines() if ln.strip()]
        header_idx = 0
        for i in range(min(6, len(lines))):
            if "Image Name" in lines[i] and "PID" in lines[i]:
                header_idx = i
                break
        data_lines = lines[header_idx+1:]
        for ln in data_lines:
            try:
                image = ln[0:25].strip()
                pid = ln[25:35].strip()
                session_name = ln[35:57].strip()
                session_num = ln[57:65].strip()
                mem = ln[65:].strip()
                mb_val, mb_str = parse_mem_to_mb(mem)
                parsed.append({
                    "image": image,
                    "pid": pid,
                    "session_name": session_name,
                    "session_num": session_num,
                    "mem_raw": mem,
                    "mem_mb": mb_val,
                    "mem_str": mb_str,
                    "cpu_pct": 0.0,
                    "disk": "0 MB/s",
                    "net": "0 Mbps"
                })
            except Exception:
                tokens = [t for t in ln.split() if t]
                if len(tokens) >= 2:
                    image = tokens[0]
                    pid = tokens[1]
                    mem = tokens[-2] + " " + tokens[-1] if len(tokens) >= 4 else tokens[-1]
                    mb_val, mb_str = parse_mem_to_mb(mem)
                    parsed.append({
                        "image": image, "pid": pid, "session_name": "", "session_num": "",
                        "mem_raw": mem, "mem_mb": mb_val, "mem_str": mb_str,
                        "cpu_pct": 0.0, "disk": "0 MB/s", "net": "0 Mbps"
                    })
    return parsed

# ---------------- memory parse helper (kept stable) ----------------
def parse_mem_to_mb(mem_str):
    if not mem_str:
        return 0.0, "0 MB"
    s = mem_str.strip().replace(",", "").upper()
    try:
        # handle "4096 K" or "4096K"
        if s.endswith("K"):
            num = float(s[:-1].strip())
            return num / 1024.0, f"{num/1024.0:.2f} MB"
        if s.endswith("MB"):
            num = float(s[:-2].strip())
            return num, f"{num:.2f} MB"
        if s.endswith("B"):
            num = float(s[:-1].strip())
            return num / (1024.0**2), f"{num/(1024.0**2):.2f} MB"
        num = float(s)
        if num > 1000:
            # assume KB
            return num / 1024.0, f"{num/1024.0:.2f} MB"
        return num, f"{num:.2f} MB"
    except:
        return 0.0, mem_str

# ---------------- WMIC perf sampling & averaging (improved) ----------------
def sample_wmic_perf_formatted():
    """
    Read WMIC perf formatted CSV once. Return mapping PID->PercentProcessorTime (as float).
    """
    out = run_cmd('wmic path Win32_PerfFormattedData_PerfProc_Process get IDProcess,PercentProcessorTime /format:csv')
    mapping = {}
    if not out:
        return mapping
    lines = [ln for ln in out.splitlines() if ln.strip()]
    for ln in lines:
        if ln.strip().startswith("Node"):
            continue
        parts = ln.split(',')
        # Typically: Node,Name,IDProcess,PercentProcessorTime
        # We scan for numeric tokens: PID and CPU value
        pid = None
        cpu = None
        for p in parts:
            ps = p.strip()
            if ps.isdigit() and pid is None:
                pid = ps
            else:
                try:
                    cpu = float(ps)
                except:
                    continue
        if pid:
            mapping[pid] = float(cpu) if cpu is not None else 0.0
    return mapping

def get_per_process_cpu_wmic_avg(sample_delay_ms=SAMPLE_DELAY_MS, samples=3):
    """
    Take multiple WMIC samples (samples), spaced by sample_delay_ms, then compute
    averaged PercentProcessorTime per PID and normalize by logical CPU count.
    Returns dict PID->normalized_cpu_pct (0..100)
    """
    sample_maps = []
    for i in range(samples):
        m = sample_wmic_perf_formatted()
        sample_maps.append(m)
        if i != samples - 1:
            time.sleep(sample_delay_ms / 1000.0)
    # keys union
    keys = set()
    for m in sample_maps:
        keys |= set(m.keys())
    cpu_count = os.cpu_count() or 1
    combined = {}
    for k in keys:
        # average across samples (ignore missing values treated as 0)
        s = 0.0
        for m in sample_maps:
            s += float(m.get(k, 0.0))
        avg = s / float(len(sample_maps))
        # normalize by logical CPU count to produce percent of total
        normalized = avg / cpu_count
        # clamp
        normalized = max(0.0, min(normalized, 100.0))
        combined[k] = normalized
    return combined

# per-process EMA smoothing store
_PERPROC_EMA = {}

def ema_smooth(pid, new_val, alpha=CPU_EMA_ALPHA):
    """
    Apply simple EMA smoothing per pid.
    """
    key = str(pid)
    prev = _PERPROC_EMA.get(key)
    if prev is None:
        _PERPROC_EMA[key] = new_val
        return new_val
    sm = alpha * new_val + (1 - alpha) * prev
    _PERPROC_EMA[key] = sm
    return sm

# ---------------- executable path helpers (WMIC then PowerShell fallback) ----------------
def get_executable_path(pid):
    """
    Try WMIC first (fast), then PowerShell Get-CimInstance fallback for better results.
    Return empty string if not found or permission denied.
    """
    pid = str(pid)
    # WMIC
    text = run_cmd(f'wmic process where ProcessId={pid} get ExecutablePath /value')
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip().lower() == "executablepath":
                val = v.strip()
                if val:
                    return val
    # fallback to PowerShell Get-CimInstance (more reliable on some Windows)
    # Use powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"ProcessId=1234\").ExecutablePath"
    try:
        # build command carefully
        ps_cmd = f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}").ExecutablePath'
        out = run_cmd(f'powershell -NoProfile -Command "{ps_cmd}"')
        out = out.strip()
        # sometimes returns blank lines; pick first non-empty
        for ln in out.splitlines():
            ln = ln.strip()
            if ln:
                return ln
    except Exception:
        pass
    return ""

# ---------------- kill process safe helper ----------------
def kill_process(pid):
    """
    Try taskkill /F first; if fails, try PowerShell Stop-Process -Id pid -Force
    Return combined output or error message.
    """
    pid = str(pid)
    out = run_cmd(f'taskkill /PID {pid} /F')
    if out and ("SUCCESS" in out.upper() or "TERMINATED" in out.upper() or "SUCCESS" in out):
        return out
    # fallback via PowerShell
    try:
        ps_cmd = f'Try {{ Stop-Process -Id {pid} -Force -ErrorAction Stop; Write-Output "PS:OK" }} Catch {{ Write-Output "PSERR:$($_.Exception.Message)" }}'
        out2 = run_cmd(f'powershell -NoProfile -Command "{ps_cmd}"')
        return (out + "\n" + out2).strip()
    except Exception as e:
        return (out + "\nERR:" + str(e)).strip()

# ---------------- mem color (unchanged) ----------------
def mem_to_color(value, max_val):
    if max_val <= 0:
        return "#2b2b2b"
    ratio = min(max(value / max_val, 0.0), 1.0)
    base = (34, 34, 34)
    red = (180, 30, 30)
    r = int(base[0] + (red[0]-base[0]) * ratio)
    g = int(base[1] + (red[1]-base[1]) * ratio)
    b = int(base[2] + (red[2]-base[2]) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------------- GUI ----------------
class FinalMonitorApp:
    def __init__(self, root):
        self.root = root
        root.title("Process Monitor — PRO Accurate (WMIC + Enhancements)")
        root.geometry("1000x760")
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass

        # Colors / theme (kept similar to original but slightly refined)
        self.bg_color = "#101214"
        self.panel_color = "#1c1f23"
        self.fg_color = "#e6e6e6"
        self.accent = "#4da6ff"
        root.configure(bg=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TButton", background=self.panel_color, foreground=self.fg_color)
        self.style.configure("Treeview", background=self.panel_color, fieldbackground=self.panel_color, foreground=self.fg_color, rowheight=22)
        self.style.map("Treeview", background=[('selected', '#2a68a0')], foreground=[('selected', 'white')])
        self.style.configure("Treeview.Heading", background="#14161a", foreground=self.fg_color, font=("Segoe UI", 10, "bold"))
        self.style.configure("TProgressbar", troughcolor=self.panel_color, background=self.accent)

        top = ttk.Frame(root, padding=8)
        top.pack(side="top", fill="x")
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(top, width=360)
        right.pack(side="right", fill="y")

        # header with drawn logo + stats (modernized spacing)
        header = ttk.Frame(left)
        header.pack(side="top", fill="x")

        logo_canvas = tk.Canvas(header, width=56, height=36, bg=self.bg_color, highlightthickness=0)
        logo_canvas.pack(side="left", padx=(0,14))
        self._draw_logo(logo_canvas)

        stats = ttk.Frame(header)
        stats.pack(side="left", anchor="center")
        self.cpu_var = tk.StringVar(value="CPU: N/A")
        self.ram_var = tk.StringVar(value="RAM: N/A")
        ttk.Label(stats, textvariable=self.cpu_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(stats, textvariable=self.ram_var, font=("Segoe UI", 10)).pack(anchor="w")

        # graph canvas (double-buffer style improvements)
        self.graph = tk.Canvas(left, height=220, bg="#060607", highlightthickness=0)
        self.graph.pack(side="top", fill="x", padx=6, pady=(8,8))
        self.cpu_history = deque(maxlen=HISTORY_LENGTH)
        self.ram_history = deque(maxlen=HISTORY_LENGTH)
        self.timestamps = deque(maxlen=HISTORY_LENGTH)

        # progress bars
        pbf = ttk.Frame(left)
        pbf.pack(side="top", fill="x", padx=6)
        ttk.Label(pbf, text="CPU").grid(row=0, column=0, sticky="w")
        self.cpu_pb = ttk.Progressbar(pbf, orient="horizontal", length=420, mode="determinate")
        self.cpu_pb.grid(row=0, column=1, padx=(8,8))
        ttk.Label(pbf, text="RAM").grid(row=1, column=0, sticky="w")
        self.ram_pb = ttk.Progressbar(pbf, orient="horizontal", length=420, mode="determinate")
        self.ram_pb.grid(row=1, column=1, padx=(8,8))

        # right controls
        ttk.Button(right, text="Refresh Now", command=self.refresh_once).pack(fill="x", pady=6, padx=8)
        self.pause_btn = ttk.Button(right, text="Pause Graph", command=self.toggle_pause)
        self.pause_btn.pack(fill="x", pady=6, padx=8)
        self.auto_refresh = True
        self.refresh_ms_var = tk.IntVar(value=REFRESH_MS)
        self.refresh_label_var = tk.StringVar(value=f"Auto-Refresh: {self.refresh_ms_var.get()} ms")
        self.auto_btn = ttk.Button(right, text="Pause Auto-Refresh", command=self.toggle_auto_refresh)
        self.auto_btn.pack(fill="x", pady=6, padx=8)
        ttk.Label(right, textvariable=self.refresh_label_var).pack(fill="x", padx=8)
        self.refresh_scale = ttk.Scale(right, from_=500, to=5000, orient="horizontal", command=lambda v: self._update_refresh_label(v))
        self.refresh_scale.set(self.refresh_ms_var.get())
        self.refresh_scale.pack(fill="x", pady=6, padx=8)

        ttk.Button(right, text="Save Graph (PS)", command=self.save_graph).pack(fill="x", pady=6, padx=8)
        ttk.Button(right, text="Export CSV", command=self.export_csv).pack(fill="x", pady=6, padx=8)
        ttk.Button(right, text="Export Process CSV", command=self.export_process_csv).pack(fill="x", pady=6, padx=8)
        ttk.Button(right, text="Exit", command=root.quit).pack(fill="x", pady=16, padx=8)

        # search & filters
        search_frame = ttk.Frame(root, padding=6)
        search_frame.pack(side="top", fill="x")
        ttk.Label(search_frame, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self.apply_search())
        ttk.Entry(search_frame, textvariable=self.search_var).pack(side="left", fill="x", expand=True, padx=(6,6))
        ttk.Button(search_frame, text="Refresh List", command=self.refresh_once).pack(side="right")

        self.cpu_filter_var = tk.BooleanVar(value=False)
        self.cpu_thresh_var = tk.DoubleVar(value=5.0)
        self.mem_filter_var = tk.BooleanVar(value=False)
        self.mem_thresh_var = tk.DoubleVar(value=50.0)
        filt = ttk.Frame(search_frame)
        filt.pack(side="left", padx=8)
        ttk.Checkbutton(filt, text="CPU >", variable=self.cpu_filter_var, command=self.apply_search).grid(row=0, column=0, sticky="w")
        ttk.Entry(filt, width=5, textvariable=self.cpu_thresh_var).grid(row=0, column=1)
        ttk.Label(filt, text="%").grid(row=0, column=2)
        ttk.Checkbutton(filt, text="Mem >", variable=self.mem_filter_var, command=self.apply_search).grid(row=0, column=3, sticky="w", padx=(8,0))
        ttk.Entry(filt, width=7, textvariable=self.mem_thresh_var).grid(row=0, column=4)
        ttk.Label(filt, text="MB").grid(row=0, column=5)

        # process table
        cols = ("Image Name", "PID", "CPU%", "Memory (MB)", "Disk", "Network")
        self.tree = ttk.Treeview(root, columns=cols, show="headings", selectmode="browse", height=18)
        for col in cols:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by_column(c))
            if col == "Image Name":
                self.tree.column(col, width=420, anchor="w")
            elif col == "PID":
                self.tree.column(col, width=70, anchor="center")
            elif col == "CPU%":
                self.tree.column(col, width=80, anchor="center")
            elif col == "Memory (MB)":
                self.tree.column(col, width=120, anchor="e")
            else:
                self.tree.column(col, width=120, anchor="center")
        self.tree.pack(side="top", fill="both", expand=True, padx=6, pady=(6,6))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.update_details_from_selection())

        # details panel
        details = ttk.Frame(right, padding=8)
        details.pack(side="bottom", fill="x")
        ttk.Label(details, text="Selected Process").pack(anchor="w")
        self.sel_image_var = tk.StringVar(value="")
        self.sel_pid_var = tk.StringVar(value="")
        self.sel_cpu_var = tk.StringVar(value="")
        self.sel_mem_var = tk.StringVar(value="")
        self.sel_path_var = tk.StringVar(value="")
        ttk.Label(details, textvariable=self.sel_image_var).pack(anchor="w")
        ttk.Label(details, textvariable=self.sel_pid_var).pack(anchor="w")
        ttk.Label(details, textvariable=self.sel_cpu_var).pack(anchor="w")
        ttk.Label(details, textvariable=self.sel_mem_var).pack(anchor="w")
        ttk.Label(details, textvariable=self.sel_path_var, wraplength=320).pack(anchor="w")
        ttk.Button(details, text="Open Location", command=self.open_selected_location).pack(fill="x", pady=(6,2))
        ttk.Button(details, text="Copy Path", command=self.copy_selected_path).pack(fill="x", pady=2)
        ttk.Button(details, text="Copy PID", command=self.copy_selected_pid).pack(fill="x", pady=2)

        bottom = ttk.Frame(root, padding=8)
        bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Kill Selected Process", command=self.kill_selected).pack(side="left")
        ttk.Label(bottom, text="Tip: Run as Administrator to terminate protected processes").pack(side="right")

        # state
        self._all_rows = []
        self._current_sort_col = None
        self._current_sort_rev = False
        self.pause = False

        # start refresh
        self.root.after(200, self.refresh_once)
        self.root.after(REFRESH_MS, self._periodic_update)

    def _draw_logo(self, canvas):
        # simple chip + pulse icon drawn on canvas (refined colors)
        w = 56; h = 36
        canvas.create_rectangle(6,8,w-6,h-8, fill="#1b1b1d", outline="#2c2c2e")
        for x in (10,18,26,34,42):
            canvas.create_line(x, h-8, x, h+2, fill="#222")
            canvas.create_line(x, 0, x, 6, fill="#222")
        canvas.create_rectangle(14,12,w-14,h-12, fill=self.accent, outline="#083a66")
        canvas.create_line(16, h/2, 22, h/2, fill="#fff", width=2)
        canvas.create_line(22, h/2, 26, h/2 - 8, fill="#fff", width=2)
        canvas.create_line(26, h/2 - 8, 30, h/2 + 8, fill="#fff", width=2)
        canvas.create_line(30, h/2 + 8, 38, h/2, fill="#fff", width=2)

    def _periodic_update(self):
        if self.auto_refresh:
            self.refresh_once()
        # dynamic refresh period from slider
        ms = max(200, int(self.refresh_ms_var.get()))
        self.refresh_label_var.set(f"Auto-Refresh: {ms} ms")
        self.root.after(ms, self._periodic_update)

    def refresh_once(self):
        t = threading.Thread(target=self._refresh_thread, daemon=True)
        t.start()

    def _refresh_thread(self):
        # gather overall numbers (CPU, RAM)
        cpu = None
        used_mb = total_mb = None
        ram_pct = None
        try:
            cpu = self._get_cpu_loadpercent_safe()
        except Exception:
            cpu = get_cpu_loadpercent()
        try:
            used_mb, total_mb, ram_pct = get_memory_usage_full()
        except Exception:
            used_mb, total_mb, ram_pct = get_memory_usage_full()
        # process list using robust csv parsing
        procs = get_process_list_raw()
        # per-process CPU% averaged then normalized by cpu_count with multiple samples
        pid_cpu_map = get_per_process_cpu_wmic_avg(sample_delay_ms=SAMPLE_DELAY_MS, samples=3)
        # apply EMA smoothing and attach to procs
        for p in procs:
            pid = p.get("pid", "")
            raw_cpu = float(pid_cpu_map.get(pid, 0.0))
            smooth_cpu = ema_smooth(pid, raw_cpu)
            p["cpu_pct"] = smooth_cpu
        ts = time.strftime("%H:%M:%S")
        # history (append)
        if cpu is not None and ram_pct is not None:
            # clamp cpu to [0,100]
            cpu = max(0.0, min(100.0, float(cpu)))
            ram_pct = max(0.0, min(100.0, float(ram_pct)))
            self.cpu_history.append(cpu)
            self.ram_history.append(ram_pct)
            self.timestamps.append(ts)
        else:
            self.cpu_history.append(0.0)
            self.ram_history.append(0.0)
            self.timestamps.append(ts)
        # ensure history length
        while len(self.cpu_history) > HISTORY_LENGTH:
            self.cpu_history.popleft()
            self.ram_history.popleft()
            self.timestamps.popleft()
        # update UI on main thread
        self.root.after(0, self._update_ui, cpu, used_mb, total_mb, ram_pct, procs)

    def _get_cpu_loadpercent_safe(self):
        """
        Safer wrapper for overall CPU load via WMIC with retry+timeout.
        Fallback: uses wmic cpu get loadpercentage OR reads using typeperf (if available).
        """
        # Prefer WMIC
        out = run_cmd_raw("wmic cpu get loadpercentage", timeout=WMIC_TIMEOUT)
        parts = [t.strip() for t in out.split() if t.strip().isdigit()]
        if parts:
            try:
                return float(parts[-1])
            except:
                pass
        # fallback to typeperf for overall CPU (if available)
        try:
            # use typeperf once with sample: typeperf "\Processor(_Total)\% Processor Time" -sc 1
            out2 = run_cmd_raw(r'typeperf "\Processor(_Total)\% Processor Time" -sc 1', timeout=WMIC_TIMEOUT)
            # parse last numeric in output
            for ln in out2.splitlines()[::-1]:
                if '"' in ln and ',' in ln:
                    # line: "timestamp","value"
                    try:
                        val = ln.split(',')[-1].strip().strip('"')
                        return float(val)
                    except:
                        continue
            # final fallback: return 0
        except Exception:
            pass
        return None

    def _update_ui(self, cpu, used_mb, total_mb, ram_pct, procs):
        # update overall labels and progress bars
        if cpu is not None:
            self.cpu_var.set(f"CPU: {cpu:.1f}%")
            try:
                self.cpu_pb['value'] = cpu
            except:
                pass
        else:
            self.cpu_var.set("CPU: N/A")
        if used_mb is not None and total_mb is not None:
            try:
                used_gb = used_mb / 1024.0
                total_gb = total_mb / 1024.0
                self.ram_var.set(f"RAM: {used_gb:.2f} GB used / {total_gb:.2f} GB total ({ram_pct:.2f}%)")
                self.ram_pb['value'] = ram_pct if ram_pct is not None else 0
            except:
                self.ram_var.set("RAM: N/A")
        else:
            self.ram_var.set("RAM: N/A")

        # draw graph (with smoothing) if not paused
        if not self.pause:
            try:
                self.draw_graph()
            except Exception:
                # protect against drawing errors
                pass

        # process rows - preserve selection
        sel = self.tree.selection()
        sel_pid = None
        if sel:
            try:
                sel_pid = self.tree.item(sel[0])["values"][1]
            except:
                sel_pid = None

        # store rows then apply search+sort via apply_search
        self._all_rows = procs
        self.apply_search()

        # restore selection
        if sel_pid:
            for iid in self.tree.get_children():
                try:
                    if str(self.tree.item(iid)["values"][1]) == str(sel_pid):
                        self.tree.selection_set(iid)
                        self.tree.see(iid)
                        break
                except:
                    pass

        # update details panel for currently selected
        self.update_details_from_selection()

        # ensure canvas shown (force idle update)
        try:
            self.graph.update_idletasks()
        except:
            pass

    def draw_graph(self):
        """
        Draw smoothed CPU & RAM history. Use moving average smoothing to reduce noise.
        """
        c = self.graph
        c.delete("all")
        w = c.winfo_width() or c.winfo_reqwidth() or 800
        h = c.winfo_height() or 220

        # background
        c.create_rectangle(0, 0, w, h, fill="#050506", outline="")

        # horizontal grid lines + labels
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(h - 28 - frac * (h - 56))
            c.create_line(10, y, w-10, y, fill="#151517")
            c.create_text(36, y, text=f"{int(frac*100)}%", fill="#3f3f3f", font=("Segoe UI", 8))

        # smoothing: moving average
        def moving_avg(seq, window=GRAPH_SMOOTH_WINDOW):
            if not seq:
                return []
            if window <= 1:
                return list(seq)
            res = []
            arr = list(seq)
            for i in range(len(arr)):
                start = max(0, i - window + 1)
                res.append(sum(arr[start:i+1]) / float(i - start + 1))
            return res

        cpu_sm = moving_avg(list(self.cpu_history), window=GRAPH_SMOOTH_WINDOW)
        ram_sm = moving_avg(list(self.ram_history), window=GRAPH_SMOOTH_WINDOW)
        N = HISTORY_LENGTH
        # pad to length N for consistent x spacing
        def pad_to(arr, n):
            if len(arr) >= n:
                return arr[-n:]
            else:
                pad = [0.0] * (n - len(arr))
                return pad + arr
        cpu_plot = pad_to(cpu_sm, N)
        ram_plot = pad_to(ram_sm, N)

        x0 = 60
        x1 = w - 24
        span = x1 - x0
        step = span / max(N - 1, 1)
        top = 20
        bottom = h - 20

        def plot_series(data, color, width=2):
            pts = []
            for i, val in enumerate(data):
                x = x0 + i * step
                y = bottom - (val / 100.0) * (bottom - top)
                pts.append((x, y))
            # lines
            for i in range(len(pts) - 1):
                c.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill=color, width=width, smooth=True)
            # small dots for latest  points
            for (x, y) in pts:
                c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill=color, outline=color)

        plot_series(cpu_plot, self.accent, width=2)
        plot_series(ram_plot, "#66d19e", width=2)

        # legend box
        c.create_rectangle(w - 220, 8, w - 22, 44, fill="#0f0f10", outline="#222")
        c.create_text(w - 200, 18, text="CPU", fill=self.accent, anchor="w", font=("Segoe UI", 9, "bold"))
        c.create_text(w - 140, 18, text="RAM", fill="#66d19e", anchor="w", font=("Segoe UI", 9, "bold"))
        # current values
        if self.cpu_history:
            c.create_text(w - 200, 34, text=f"{self.cpu_history[-1]:.1f}%", fill=self.fg_color, anchor="w", font=("Segoe UI", 8))
        if self.ram_history:
            c.create_text(w - 140, 34, text=f"{self.ram_history[-1]:.1f}%", fill=self.fg_color, anchor="w", font=("Segoe UI", 8))

        # force a small update so canvas actually renders on some GPUs
        try:
            self.graph.update_idletasks()
        except:
            pass

    def apply_search(self):
        q = self.search_var.get().lower().strip()
        rows = list(self._all_rows)
        # sorting
        if self._current_sort_col:
            col = self._current_sort_col
            rev = self._current_sort_rev
            def keyfunc(p):
                if col == "Image Name":
                    return p["image"].lower()
                if col == "PID":
                    try:
                        return int(p["pid"])
                    except:
                        return 0
                if col == "CPU%":
                    try:
                        return float(p["cpu_pct"])
                    except:
                        return 0.0
                if col == "Memory (MB)":
                    return float(p["mem_mb"]) if isinstance(p["mem_mb"], (int, float)) else 0.0
                return p.get("disk","")
            rows.sort(key=keyfunc, reverse=rev)

        cpu_filter = self.cpu_filter_var.get()
        mem_filter = self.mem_filter_var.get()
        try:
            cpu_thresh = float(self.cpu_thresh_var.get())
        except:
            cpu_thresh = 0.0
        try:
            mem_thresh = float(self.mem_thresh_var.get())
        except:
            mem_thresh = 0.0

        self.tree.delete(*self.tree.get_children())
        max_mem_all = max((x["mem_mb"] for x in rows), default=1.0)
        for p in rows:
            if q and not (q in p["image"].lower() or q in p["pid"] or q in p["mem_str"].lower()):
                continue
            if cpu_filter:
                try:
                    if float(p["cpu_pct"]) <= cpu_thresh:
                        continue
                except:
                    continue
            if mem_filter:
                try:
                    if float(p["mem_mb"]) <= mem_thresh:
                        continue
                except:
                    continue
            mem_mb = p["mem_mb"]
            cpu_pct = f"{p['cpu_pct']:.1f}" if isinstance(p['cpu_pct'], float) else str(p['cpu_pct'])
            vals = (p["image"], p["pid"], cpu_pct, f"{mem_mb:.2f}", p.get("disk","0 MB/s"), p.get("net","0 Mbps"))
            iid = self.tree.insert("", "end", values=vals)
            color = mem_to_color(mem_mb, max_mem_all)
            tagname = f"mem_{int(mem_mb)}"
            if not tagname in self.tree.tag_has(tagname):
                self.tree.tag_configure(tagname, background=color)
            self.tree.item(iid, tags=(tagname,))

    def toggle_pause(self):
        self.pause = not self.pause
        self.pause_btn.config(text="Resume Graph" if self.pause else "Pause Graph")

    def toggle_auto_refresh(self):
        self.auto_refresh = not self.auto_refresh
        self.auto_btn.config(text="Resume Auto-Refresh" if not self.auto_refresh else "Pause Auto-Refresh")

    def _update_refresh_label(self, v):
        try:
            self.refresh_ms_var.set(int(float(v)))
        except:
            pass
        self.refresh_label_var.set(f"Auto-Refresh: {self.refresh_ms_var.get()} ms")

    def kill_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a process first.")
            return
        item = self.tree.item(sel[0])
        vals = item.get("values", [])
        if len(vals) < 2:
            messagebox.showerror("Error", "Cannot determine PID.")
            return
        pid = vals[1]
        image = vals[0]
        if not pid or not str(pid).isdigit():
            messagebox.showerror("Error", f"Invalid PID: {pid}")
            return
        if not messagebox.askyesno("Confirm", f"Kill {image} (PID {pid})?"):
            return
        out = kill_process(pid)
        messagebox.showinfo("Result", f"Attempted to kill PID {pid}.\n\n{out[:1000]}")
        # refresh immediately to reflect change
        self.refresh_once()

    def update_details_from_selection(self):
        sel = self.tree.selection()
        if not sel:
            self.sel_image_var.set("")
            self.sel_pid_var.set("")
            self.sel_cpu_var.set("")
            self.sel_mem_var.set("")
            self.sel_path_var.set("")
            return
        vals = self.tree.item(sel[0]).get("values", [])
        if len(vals) < 4:
            return
        image, pid, cpu, mem = vals[0], str(vals[1]), str(vals[2]), str(vals[3])
        path = get_executable_path(pid)
        self.sel_image_var.set(f"Name: {image}")
        self.sel_pid_var.set(f"PID: {pid}")
        self.sel_cpu_var.set(f"CPU: {cpu}%")
        self.sel_mem_var.set(f"Mem: {mem} MB")
        self.sel_path_var.set(f"Path: {path if path else 'N/A'}")

    def open_selected_location(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0]).get("values", [])
        if len(vals) < 2:
            return
        pid = str(vals[1])
        path = get_executable_path(pid)
        if path:
            try:
                # explorer /select,"C:\path\to\file.exe"
                subprocess.Popen(f'explorer /select,"{path}"', shell=True)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showinfo("Info", "Executable path not available.")

    def copy_selected_path(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0]).get("values", [])
        if len(vals) < 2:
            return
        pid = str(vals[1])
        path = get_executable_path(pid)
        if path:
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(path)
                self.root.update()
            except:
                pass

    def copy_selected_pid(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0]).get("values", [])
        if len(vals) < 2:
            return
        pid = str(vals[1])
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(pid)
            self.root.update()
        except:
            pass

    def save_graph(self):
        f = filedialog.asksaveasfilename(defaultextension=".ps", filetypes=[("PostScript","*.ps"),("All files","*.*")])
        if not f:
            return
        try:
            self.graph.postscript(file=f)
            messagebox.showinfo("Saved", f"Graph saved to {f}. Convert to PNG if needed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_csv(self):
        if not self.timestamps:
            messagebox.showinfo("No Data", "No history yet")
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv"),("All files","*.*")])
        if not f:
            return
        try:
            with open(f, "w", newline="") as csvf:
                w = csv.writer(csvf)
                w.writerow(["timestamp","cpu_percent","ram_percent"])
                for t,c,r in zip(self.timestamps, self.cpu_history, self.ram_history):
                    w.writerow([t, f"{c:.2f}", f"{r:.2f}"])
            messagebox.showinfo("Saved", f"History exported to {f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_process_csv(self):
        rows = list(self._all_rows)
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv"),("All files","*.*")])
        if not f:
            return
        try:
            with open(f, "w", newline="") as csvf:
                w = csv.writer(csvf)
                w.writerow(["image","pid","cpu_pct","mem_mb","path"])
                for p in rows:
                    pid = str(p.get("pid",""))
                    path = get_executable_path(pid)
                    w.writerow([p.get("image",""), pid, f"{float(p.get('cpu_pct',0.0)):.2f}", f"{float(p.get('mem_mb',0.0)):.2f}", path])
            messagebox.showinfo("Saved", f"Process list exported to {f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _sort_by_column(self, col):
        prev = self._current_sort_col
        if prev == col:
            self._current_sort_rev = not self._current_sort_rev
        else:
            self._current_sort_col = col
            self._current_sort_rev = False
        self.apply_search()

# ----------------- script entry -----------------
if __name__ == "__main__":
    if sys.platform != "win32":
        # show message then exit
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Unsupported", "This script works only on Windows.")
        sys.exit(1)
    root = tk.Tk()
    app = FinalMonitorApp(root)
    root.mainloop()
def get_memory_usage_full():
    """
    Returns (used_mb, total_mb, percent_used) using WMIC.
    More stable and works on all Windows versions.
    """
    out = run_cmd('wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value')
    free_kb = None
    total_kb = None

    for line in out.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "freephysicalmemory":
                try:
                    free_kb = int(val)
                except:
                    free_kb = None
            elif key == "totalvisiblememorysize":
                try:
                    total_kb = int(val)
                except:
                    total_kb = None

    if free_kb is None or total_kb is None or total_kb == 0:
        return None, None, None

    used_kb = total_kb - free_kb
    used_mb = used_kb / 1024.0
    total_mb = total_kb / 1024.0
    percent = (used_kb / total_kb) * 100.0

    return used_mb, total_mb, percent
