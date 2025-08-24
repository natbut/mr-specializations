import os
import shlex
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------- Platform helpers ----------

def launch_in_new_terminal(cmd_list, cwd=None, keep_open=False):
    """
    Launch cmd_list in a NEW terminal window so the process can run indefinitely.
    """
    if os.name == "nt":
        if keep_open:
            full = ["cmd.exe", "/k"] + cmd_list  # keep window open (useful for debugging)
            return subprocess.Popen(full, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        return subprocess.Popen(cmd_list, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # For Linux/macOS, adapt as needed (gnome-terminal, xterm, osascript, etc.)
        return subprocess.Popen(cmd_list, cwd=cwd)

def is_valid_int(s):
    try:
        int(s)
        return True
    except Exception:
        return False

# ---------- UI App ----------

class AgentLauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Agent Planner Launcher")
        self.geometry("980x700")

        self._build_top_controls()
        self._build_launch_controls()
        self._build_trigger_section()

        # Defaults
        self.sim_var.set(True)
        self.script_choice.set("Passenger")

        # Store created robot panels: robot_id -> {"frame":..., "role":..., ...}
        self.robot_panels = {}

    # ---- Top: Working directory + config path ----

    def _build_top_controls(self):
        top = ttk.LabelFrame(self, text="Paths")
        top.pack(fill="x", padx=10, pady=10)

        # Working directory
        wd_row = ttk.Frame(top)
        wd_row.pack(fill="x", padx=8, pady=(8,4))
        ttk.Label(wd_row, text="Working directory:").pack(side="left")
        self.working_dir = tk.StringVar(value="C:\\Users\\Nathan Butler\\Documents\\OSU\\RDML\\mr-specializations\\hardware_agents")
        ttk.Entry(wd_row, textvariable=self.working_dir, width=80).pack(side="left", padx=6)
        ttk.Button(wd_row, text="Browse...", command=self._pick_working_dir).pack(side="left")

        # Config file path
        cfg_row = ttk.Frame(top)
        cfg_row.pack(fill="x", padx=8, pady=(4,8))
        ttk.Label(cfg_row, text="Config file:").pack(side="left")
        self.config_fp = tk.StringVar(value=r"conf\\test_1.yaml")
        ttk.Entry(cfg_row, textvariable=self.config_fp, width=80).pack(side="left", padx=6)
        ttk.Button(cfg_row, text="Browse...", command=self._pick_config_file).pack(side="left")

    def _pick_working_dir(self):
        path = filedialog.askdirectory(title="Select Working Directory")
        if path:
            self.working_dir.set(path)

    def _pick_config_file(self):
        path = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML", "*.yaml;*.yml"), ("All files","*.*")]
        )
        if path:
            wd = self.working_dir.get().strip()
            if wd and os.path.commonpath([os.path.abspath(wd), os.path.abspath(path)]) == os.path.abspath(wd):
                self.config_fp.set(os.path.relpath(path, wd))
            else:
                self.config_fp.set(path)

    # ---- Middle: Agent launch controls ----

    def _build_launch_controls(self):
        lf = ttk.LabelFrame(self, text="Launch Agents")
        lf.pack(fill="x", padx=10, pady=10)

        row1 = ttk.Frame(lf)
        row1.pack(fill="x", padx=8, pady=(8,4))

        ttk.Label(row1, text="Script:").pack(side="left")
        self.script_choice = tk.StringVar(value="Passenger")
        ttk.Combobox(
            row1, textvariable=self.script_choice,
            values=["Passenger", "Mothership"], width=12, state="readonly"
        ).pack(side="left", padx=6)

        ttk.Label(row1, text="robot_id:").pack(side="left", padx=(12,2))
        self.robot_id_var = tk.StringVar(value="1")
        ttk.Entry(row1, textvariable=self.robot_id_var, width=8).pack(side="left")

        ttk.Label(row1, text="--sim_comms:").pack(side="left", padx=(12,2))
        self.sim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, variable=self.sim_var).pack(side="left")

        row2 = ttk.Frame(lf)
        row2.pack(fill="x", padx=8, pady=(4,10))

        ttk.Label(row2, text="--config_fp:").pack(side="left")
        ttk.Entry(row2, textvariable=self.config_fp, width=60).pack(side="left", padx=6)

        ttk.Button(row2, text="Launch Agent", command=self._on_launch_agent).pack(side="left", padx=(10,0))
        ttk.Button(row2, text="Add Trigger Panel for robot_id", command=self._on_add_robot_panel).pack(side="left", padx=6)

        hint = ttk.Label(
            lf, foreground="#666",
            text="Notes: Set the working directory to where HardwarePassenger.py / HardwareMothership.py / trigger.py run.\n"
                 "Each launch opens in a new terminal and runs indefinitely (your while loop)."
        )
        hint.pack(fill="x", padx=8, pady=(0,8))

    def _on_launch_agent(self):
        wd = self.working_dir.get().strip() or None
        if not wd or not os.path.isdir(wd):
            messagebox.showerror("Missing working directory", "Please set a valid working directory.")
            return

        cfg = self.config_fp.get().strip()
        if not cfg:
            messagebox.showerror("Missing config file", "Please set --config_fp (relative or absolute).")
            return

        rid = self.robot_id_var.get().strip()
        if not is_valid_int(rid):
            messagebox.showerror("Invalid robot_id", "robot_id must be an integer.")
            return

        sim_val = "True" if self.sim_var.get() else "False"
        script = "HardwarePassenger.py" if self.script_choice.get() == "Passenger" else "HardwareMothership.py"

        cmd = [
            sys.executable, script,
            "--config_fp", cfg,
            "--robot_id", str(int(rid)),
            "--sim_comms", sim_val,
        ]

        try:
            launch_in_new_terminal(cmd, cwd=wd, keep_open=True)
        except FileNotFoundError:
            messagebox.showerror("Launch error", f"Could not find script '{script}' in: {wd}")
        except Exception as e:
            messagebox.showerror("Launch error", str(e))

    def _on_add_robot_panel(self):
        rid = self.robot_id_var.get().strip()
        if not is_valid_int(rid):
            messagebox.showerror("Invalid robot_id", "robot_id must be an integer.")
            return
        rid = int(rid)
        if rid in self.robot_panels:
            messagebox.showinfo("Exists", f"A trigger panel for robot_id {rid} already exists.")
            return
        role = self.script_choice.get()  # "Passenger" or "Mothership"
        self._create_robot_trigger_panel(rid, role)

    # ---- Bottom: Trigger panels ----

    def _build_trigger_section(self):
        self.trigger_frame = ttk.LabelFrame(self, text="Robot Triggers")
        self.trigger_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(self.trigger_frame, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(self.trigger_frame, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

    def _create_robot_trigger_panel(self, robot_id: int, role: str):
        card = ttk.LabelFrame(self.inner, text=f"{role} — Robot {robot_id}")
        card.pack(fill="x", padx=8, pady=6)

        row = ttk.Frame(card)
        row.pack(fill="x", padx=8, pady=8)

        # Tile style via simple frames + buttons
        def make_tile(parent, title, help_text, btn_text, on_click, include_entry=False):
            tile = ttk.Frame(parent, relief="groove", padding=10)
            tile.pack(side="left", padx=8, pady=4, fill="x")
            ttk.Label(tile, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(tile, text=help_text, foreground="#555").pack(anchor="w", pady=(2,6))
            entry_var = None
            if include_entry:
                entry_var = tk.StringVar(value="")
                ttk.Entry(tile, textvariable=entry_var, width=40).pack(anchor="w", pady=(0,6))
            ttk.Button(tile, text=btn_text, command=lambda: on_click(entry_var.get() if entry_var else "")).pack(anchor="w")
            return entry_var

        if role == "Passenger":
            # PLAN tile (no content)
            make_tile(
                row,
                title="Plan",
                help_text="Send planning trigger",
                btn_text="Send Plan",
                on_click=lambda _content="": self._send_trigger(robot_id, "plan", "")
            )

            # UPDATE tile (content field)
            self._make_update_tile(row, robot_id)

        else:  # Mothership
            make_tile(
                row,
                title="Coordinate",
                help_text="Send coordinate trigger",
                btn_text="Send Coordinate",
                on_click=lambda _content="": self._send_trigger(robot_id, "coordinate", "")
            )

        self.robot_panels[robot_id] = {"frame": card, "role": role}

    def _make_update_tile(self, parent, robot_id: int):
        # Separate so we can validate content
        def on_update_send(content_str: str):
            # Accept arbitrary whitespace-separated numbers, but enforce numeric for update (your trigger expects floats)
            tokens = shlex.split(content_str.strip()) if content_str.strip() else []
            
            # if not tokens:
            #     messagebox.showerror("Invalid content", "Provide numeric content for Update (e.g., '44.12 -123.30').")
            #     return
            # try:
            #     floats = [float(t) for t in tokens]
            # except ValueError:
            #     messagebox.showerror("Invalid content", "Update content must be numeric (whitespace-separated).")
            #     return
            
            # If you want exactly 2 floats (lat, lon), uncomment:
            # if len(floats) != 2:
            #     messagebox.showerror("Invalid content", "For Update, provide exactly two numbers: 'lat lon'.")
            #     return
            # Pass back as the original tokens string; _send_trigger will expand to argv tokens
            self._send_trigger(robot_id, "update", content_str.strip())

        tile = ttk.Frame(parent, relief="groove", padding=10)
        tile.pack(side="left", padx=8, pady=4, fill="x")
        ttk.Label(tile, text="Update", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(tile, text="Send numeric update payload (e.g., lat lon). Must do on startup.", foreground="#555").pack(anchor="w", pady=(2,6))
        content_var = tk.StringVar(value="")
        ttk.Entry(tile, textvariable=content_var, width=40).pack(anchor="w", pady=(0,6))
        ttk.Button(tile, text="Send Update", command=lambda: on_update_send(content_var.get())).pack(anchor="w")

    def _send_trigger(self, robot_id: int, trig_type: str, content: str):
        wd = self.working_dir.get().strip() or None
        if not wd or not os.path.isdir(wd):
            messagebox.showerror("Missing working directory", "Please set a valid working directory.")
            return

        if trig_type not in {"update", "plan", "coordinate"}:
            messagebox.showerror("Invalid type", "Internal error: unknown trigger type.")
            return

        cmd = [sys.executable, "trigger.py", "--type", trig_type, "--robot_id", str(int(robot_id))]

        # For UPDATE, split content into separate argv tokens to satisfy argparse nargs='+'
        if trig_type == "update":
            tokens = shlex.split(content.strip()) if content.strip() else []
            # if not tokens:
            #     messagebox.showerror("Invalid content", "Provide numeric content for Update (e.g., '44.12 -123.30').")
            #     return
            # # Validate numeric now so trigger.py doesn't exit silently
            # try:
            #     _ = [float(t) for t in tokens]
            # except ValueError:
            #     messagebox.showerror("Invalid content", "Update content must be numeric.")
            #     return
            if len(tokens) > 0:
                cmd += ["--content"] + tokens

        # plan/coordinate have no content in your trigger.py

        try:
            launch_in_new_terminal(cmd, cwd=wd, keep_open=False)
        except FileNotFoundError:
            messagebox.showerror("Launch error", f"Could not find 'trigger.py' in: {wd}")
        except Exception as e:
            messagebox.showerror("Launch error", str(e))


if __name__ == "__main__":
    app = AgentLauncherApp()
    app.mainloop()
