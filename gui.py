import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

from math_utils import parse_function
from solvers import bisection, newton, secant, fixed_point

# ─────────────────────────────────────────────
#  Color palette & fonts
# ─────────────────────────────────────────────
BG        = "#0f0f13"
PANEL     = "#16161d"
CARD      = "#1c1c26"
BORDER    = "#2e2e42"
ACCENT    = "#7c6af7"        # violet
ACCENT2   = "#4fc4cf"        # teal
SUCCESS   = "#5dd88a"
WARNING   = "#f0a04b"
DANGER    = "#e55c5c"
TEXT      = "#e8e6f0"
TEXT_MUT  = "#7a7890"
TEXT_DIM  = "#4a4860"

METHOD_COLORS = {
    "Bisection":   "#7c6af7",
    "Newton":      "#4fc4cf",
    "Secant":      "#5dd88a",
    "Fixed-Point": "#f0a04b",
}

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEAD   = ("Courier New", 11, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_MONO   = ("Courier New", 10)
FONT_SMALL  = ("Courier New", 8)

PRESET_FUNCTIONS = [
    ("x³ − x − 2",          "x**3 - x - 2"),
    ("sin(x) − x/2",        "sin(x) - x/2"),
    ("eˣ − 3x",             "exp(x) - 3*x"),
    ("x² − 2 (√2 root)",   "x**2 - 2"),
    ("cos(x) − x",          "cos(x) - x"),
    ("Flat slope (Newton fails)", "x**3"),
]

# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def make_card(parent, **kw):
    return tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                    highlightthickness=1, **kw)

def label(parent, text, font=FONT_BODY, fg=TEXT, bg=None, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=bg or parent["bg"], **kw)

def entry(parent, width=18, **kw):
    e = tk.Entry(parent, font=FONT_MONO, bg=PANEL, fg=TEXT,
                 insertbackground=ACCENT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=width, **kw)
    e.bind("<FocusIn>",  lambda _: e.config(highlightbackground=ACCENT))
    e.bind("<FocusOut>", lambda _: e.config(highlightbackground=BORDER))
    return e

def accent_btn(parent, text, cmd, color=None, **kw):
    c = color or ACCENT
    b = tk.Button(parent, text=text, command=cmd,
                  font=FONT_HEAD, fg=BG, bg=c, activebackground=c,
                  relief="flat", cursor="hand2", padx=14, pady=6, **kw)
    b.bind("<Enter>", lambda _: b.config(bg=_lighten(c)))
    b.bind("<Leave>", lambda _: b.config(bg=c))
    return b

def ghost_btn(parent, text, cmd, **kw):
    b = tk.Button(parent, text=text, command=cmd,
                  font=FONT_BODY, fg=TEXT_MUT, bg=CARD,
                  activebackground=BORDER, relief="flat",
                  cursor="hand2", padx=8, pady=4, **kw)
    return b

def _lighten(hex_color):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    r, g, b = min(255, r + 30), min(255, g + 30), min(255, b + 30)
    return f"#{r:02x}{g:02x}{b:02x}"

# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────

class NumericalShowdown(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("The Numerical Showdown")
        self.geometry("1280x860")
        self.minsize(1100, 780)
        self.configure(bg=BG)
        self.resizable(True, True)

        self.results = {}          # method_name -> (root, iters, history, time_ms)
        self.active_methods = {m: tk.BooleanVar(value=True) for m in METHOD_COLORS}

        self._build_layout()
        self._set_defaults()

    # ── Layout ────────────────────────────────

    def _build_layout(self):
        # ── Header
        hdr = tk.Frame(self, bg=BG, pady=14)
        hdr.pack(fill="x", padx=28)
        label(hdr, "THE NUMERICAL SHOWDOWN", font=FONT_TITLE, fg=ACCENT, bg=BG).pack(side="left")
        label(hdr, "root-finding benchmark suite", font=FONT_SMALL,
              fg=TEXT_MUT, bg=BG).pack(side="left", padx=(10, 0), pady=(8, 0))

        # ── Separator
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=0)

        # ── Main body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(14, 0))
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_main(body)

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG, width=300)
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        side.pack_propagate(False)

        # ── Function input
        fc = make_card(side)
        fc.pack(fill="x", pady=(0, 10))
        _pad = dict(padx=14, pady=6)

        label(fc, "FUNCTION  f(x) = 0", font=FONT_HEAD, fg=ACCENT).pack(anchor="w", **_pad)
        tk.Frame(fc, bg=BORDER, height=1).pack(fill="x", padx=14)

        self.func_var = tk.StringVar()
        fe = entry(fc, width=28, textvariable=self.func_var)
        fe.pack(fill="x", padx=14, pady=(8, 2))

        label(fc, "presets:", font=FONT_SMALL, fg=TEXT_MUT).pack(anchor="w", padx=14)
        pf = tk.Frame(fc, bg=CARD)
        pf.pack(fill="x", padx=14, pady=(2, 10))
        for disp, expr in PRESET_FUNCTIONS:
            def _set(e=expr): self.func_var.set(e)
            ghost_btn(pf, disp, _set).pack(anchor="w", pady=1)

        # ── Parameters
        pc = make_card(side)
        pc.pack(fill="x", pady=(0, 10))
        label(pc, "PARAMETERS", font=FONT_HEAD, fg=ACCENT).pack(anchor="w", padx=14, pady=(8, 4))
        tk.Frame(pc, bg=BORDER, height=1).pack(fill="x", padx=14)

        params_frame = tk.Frame(pc, bg=CARD, pady=8)
        params_frame.pack(fill="x")

        def row(lbl, default, attr):
            r = tk.Frame(params_frame, bg=CARD)
            r.pack(fill="x", padx=14, pady=3)
            label(r, f"{lbl:<18}", font=FONT_SMALL, fg=TEXT_MUT).pack(side="left")
            v = tk.StringVar(value=default)
            setattr(self, attr, v)
            e = entry(r, width=10, textvariable=v)
            e.pack(side="right")
            return e

        row("Interval  a",    "-2",    "p_a")
        row("Interval  b",    " 2",    "p_b")
        row("Initial x₀",    " 1",    "p_x0")
        row("Initial x₁",    " 1.5",  "p_x1")
        row("Tolerance  ε",  "1e-6",  "p_eps")
        row("α (fixed-point)", "-0.2", "p_alpha")
        row("Max iterations", "100",  "p_maxiter")

        # ── Method toggles
        mc = make_card(side)
        mc.pack(fill="x", pady=(0, 10))
        label(mc, "METHODS", font=FONT_HEAD, fg=ACCENT).pack(anchor="w", padx=14, pady=(8, 4))
        tk.Frame(mc, bg=BORDER, height=1).pack(fill="x", padx=14)

        mf = tk.Frame(mc, bg=CARD, pady=6)
        mf.pack(fill="x")
        for method, var in self.active_methods.items():
            row_f = tk.Frame(mf, bg=CARD)
            row_f.pack(fill="x", padx=14, pady=2)
            cb = tk.Checkbutton(row_f, variable=var, bg=CARD,
                                fg=METHOD_COLORS[method], activebackground=CARD,
                                selectcolor=CARD, relief="flat")
            cb.pack(side="left")
            dot = tk.Frame(row_f, bg=METHOD_COLORS[method], width=8, height=8)
            dot.pack(side="left", padx=(0, 6))
            label(row_f, method, font=FONT_BODY, fg=METHOD_COLORS[method]).pack(side="left")

        # ── Run button
        accent_btn(side, "▶  RUN SHOWDOWN", self._run).pack(fill="x", pady=(4, 8))

    def _build_main(self, parent):
        main = tk.Frame(parent, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        # ── Stat cards
        stats_row = tk.Frame(main, bg=BG)
        stats_row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        stats_row.columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_cards = {}
        for i, method in enumerate(METHOD_COLORS):
            c = make_card(stats_row, padx=10, pady=8)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            col = METHOD_COLORS[method]
            label(c, method.upper(), font=FONT_SMALL, fg=col).pack(anchor="w")
            val_lbl = label(c, "—", font=("Courier New", 16, "bold"), fg=col)
            val_lbl.pack(anchor="w")
            sub_lbl = label(c, "awaiting run", font=FONT_SMALL, fg=TEXT_DIM)
            sub_lbl.pack(anchor="w")
            self.stat_cards[method] = (val_lbl, sub_lbl)

        # ── Notebook (tabs)
        nb_frame = tk.Frame(main, bg=BG)
        nb_frame.grid(row=1, column=0, sticky="nsew")
        nb_frame.rowconfigure(0, weight=1)
        nb_frame.columnconfigure(0, weight=1)

        self.tab_btns = {}
        self.tab_frames = {}

        tab_bar = tk.Frame(nb_frame, bg=BG)
        tab_bar.pack(fill="x")

        self.tab_content = tk.Frame(nb_frame, bg=PANEL,
                                     highlightbackground=BORDER,
                                     highlightthickness=1)
        self.tab_content.pack(fill="both", expand=True)

        for tname in ("Convergence Plot", "Function Plot", "Report"):
            frame = tk.Frame(self.tab_content, bg=PANEL)
            self.tab_frames[tname] = frame
            btn = tk.Button(tab_bar, text=tname, font=FONT_BODY,
                            bg=BG, fg=TEXT_MUT, relief="flat",
                            cursor="hand2", padx=14, pady=6,
                            command=lambda n=tname: self._show_tab(n))
            btn.pack(side="left")
            self.tab_btns[tname] = btn

        # Build plot areas
        self._build_convergence_tab(self.tab_frames["Convergence Plot"])
        self._build_function_tab(self.tab_frames["Function Plot"])
        self._build_report_tab(self.tab_frames["Report"])

        self._show_tab("Convergence Plot")

    def _show_tab(self, name):
        for n, f in self.tab_frames.items():
            f.pack_forget()
            self.tab_btns[n].config(fg=TEXT_MUT, bg=BG)
        self.tab_frames[name].pack(fill="both", expand=True)
        self.tab_btns[name].config(fg=ACCENT, bg=CARD)

    # ── Plot tabs ─────────────────────────────

    def _make_figure(self):
        fig = plt.Figure(facecolor=PANEL, edgecolor=PANEL)
        return fig

    def _style_ax(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT_MUT, labelsize=8)
        ax.spines[:].set_color(BORDER)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(10)
        ax.title.set_fontfamily("Courier New")
        if title:  ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel, color=TEXT_MUT, fontsize=8, fontfamily="Courier New")
        if ylabel: ax.set_ylabel(ylabel, color=TEXT_MUT, fontsize=8, fontfamily="Courier New")
        ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.7)

    def _embed_fig(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def _build_convergence_tab(self, frame):
        self.conv_fig = self._make_figure()
        # Create a 1x2 grid for the two requested plots
        self.ax_iter = self.conv_fig.add_subplot(121)
        self.ax_time = self.conv_fig.add_subplot(122)
        
        self.conv_fig.subplots_adjust(wspace=0.3, bottom=0.15) # Add space between them
        self.conv_canvas = self._embed_fig(self.conv_fig, frame)

    def _build_function_tab(self, frame):
        self.func_fig = self._make_figure()
        self.func_ax  = self.func_fig.add_subplot(111)
        self._style_ax(self.func_ax, title="f(x) with roots", xlabel="x", ylabel="f(x)")
        self.func_canvas = self._embed_fig(self.func_fig, frame)

    def _build_report_tab(self, frame):
        txt_frame = tk.Frame(frame, bg=PANEL)
        txt_frame.pack(fill="both", expand=True, padx=14, pady=14)

        self.report_text = tk.Text(
            txt_frame, font=FONT_MONO, bg=CARD, fg=TEXT,
            insertbackground=ACCENT, relief="flat",
            highlightbackground=BORDER, highlightthickness=1,
            wrap="word", state="disabled", padx=12, pady=10,
        )
        sb = ttk.Scrollbar(txt_frame, command=self.report_text.yview)
        self.report_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.report_text.pack(fill="both", expand=True)

        # colour tags
        self.report_text.tag_config("header",  foreground=ACCENT,   font=("Courier New", 11, "bold"))
        self.report_text.tag_config("sub",      foreground=ACCENT2,  font=("Courier New", 10, "bold"))
        self.report_text.tag_config("win",      foreground=SUCCESS,  font=FONT_MONO)
        self.report_text.tag_config("warn",     foreground=WARNING,  font=FONT_MONO)
        self.report_text.tag_config("fail",     foreground=DANGER,   font=FONT_MONO)
        self.report_text.tag_config("muted",    foreground=TEXT_MUT, font=FONT_SMALL)
        self.report_text.tag_config("normal",   foreground=TEXT,     font=FONT_MONO)
        for method, col in METHOD_COLORS.items():
            self.report_text.tag_config(method, foreground=col, font=("Courier New", 10, "bold"))

    # ── Defaults ──────────────────────────────

    def _set_defaults(self):
        self.func_var.set("x**3 - x - 2")

    # ── Core run logic ────────────────────────

    def _run(self):
        try:
            f       = parse_function(self.func_var.get().strip())
            a       = float(self.p_a.get())
            b       = float(self.p_b.get())
            x0      = float(self.p_x0.get())
            x1      = float(self.p_x1.get())
            eps     = float(self.p_eps.get())
            alpha   = float(self.p_alpha.get())
            maxiter = int(self.p_maxiter.get())
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self.results = {}
        runners = {
            "Bisection":   lambda: bisection(f, a, b, eps, maxiter),
            "Newton":      lambda: newton(f, x0, eps, maxiter),
            "Secant":      lambda: secant(f, x0, x1, eps, maxiter),
            "Fixed-Point": lambda: fixed_point(f, x0, alpha, eps, maxiter),
        }

        for method, run in runners.items():
            if not self.active_methods[method].get():
                continue
            t0 = time.perf_counter()
            try:
                # UNPACK 4 VARIABLES NOW
                root, iters, history, t_history = run()
            except Exception as e:
                root, iters, history, t_history = None, 0, [], []
            ms = (time.perf_counter() - t0) * 1000
            
            # Save t_history into the results dictionary
            self.results[method] = (root, iters, history, t_history, ms)

        self._update_stat_cards()
        self._draw_convergence()
        self._draw_function(f, a, b)
        self._write_report(f)

    def _update_stat_cards(self):
        min_iters = None
        for m, (root, iters, history, t_history, _) in self.results.items():
            if root is not None and (min_iters is None or iters < min_iters):
                min_iters = iters

        for method, (val_lbl, sub_lbl) in self.stat_cards.items():
            if method not in self.results:
                val_lbl.config(text="—")
                sub_lbl.config(text="not run", fg=TEXT_DIM)
                continue
            
            root, iters, _, _, ms = self.results[method]
            if root is None:
                val_lbl.config(text="FAIL")
                sub_lbl.config(text="did not converge", fg=DANGER)
            else:
                val_lbl.config(text=f"{root:.6f}")
                badge = f"{iters} iters · {ms:.2f}ms"
                fg = SUCCESS if iters == min_iters else TEXT_DIM
                sub_lbl.config(text=badge, fg=fg)

    def _draw_convergence(self):
        # Clear both axes
        self.ax_iter.clear()
        self.ax_time.clear()
        
        self._style_ax(self.ax_iter, title="Error vs Iterations", xlabel="Iteration", ylabel="log₁₀(error)")
        self._style_ax(self.ax_time, title="Error vs Time (ms)", xlabel="Time (ms)", ylabel="")
        
        plotted = False
        for method, (root, iters, history, t_history, _) in self.results.items():
            if root is None or not history:
                continue
            
            safe = [h for h in history if h > 0]
            if safe:
                log_err = np.log10(safe)
                
                # Plot 1: Error vs Iterations
                self.ax_iter.plot(range(1, len(log_err) + 1), log_err,
                        color=METHOD_COLORS[method], linewidth=1.8, marker="o", markersize=3.5, label=method)
                
                # Plot 2: Error vs Time
                # Ensure t_history matches the length of safe log_err
                safe_time = t_history[:len(log_err)] 
                self.ax_time.plot(safe_time, log_err,
                        color=METHOD_COLORS[method], linewidth=1.8, marker="s", markersize=3.5, label=method)
                
                plotted = True
                
        if plotted:
            self.ax_iter.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, prop={"family": "Courier New"})
            self.ax_time.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, prop={"family": "Courier New"})
            
        self.conv_canvas.draw()

    def _draw_function(self, f, a, b):
        ax = self.func_ax
        ax.clear()
        self._style_ax(ax, title="f(x) with found roots", xlabel="x", ylabel="f(x)")

        margin = (b - a) * 0.3
        xs = np.linspace(a - margin, b + margin, 600)
        try:
            ys = np.array([f(x) for x in xs], dtype=float)
        except Exception:
            self.func_canvas.draw()
            return

        ax.plot(xs, ys, color=ACCENT2, linewidth=1.8, label="f(x)")
        ax.axhline(0, color=BORDER, linewidth=0.8, linestyle="--")

        for method, (root, iters, _, _) in self.results.items():
            if root is None:
                continue
            try:
                fy = f(root)
            except Exception:
                continue
            ax.plot(root, fy, "o", color=METHOD_COLORS[method],
                    markersize=9, label=f"{method} root ≈ {root:.4f}",
                    zorder=5)

        ax.legend(facecolor=CARD, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=8,
                  prop={"family": "Courier New"})
        self.func_canvas.draw()

    def _write_report(self, f):
        rt = self.report_text
        rt.config(state="normal")
        rt.delete("1.0", "end")

        def w(text, tag="normal"):
            rt.insert("end", text, tag)

        w("╔══════════════════════════════════════════════════════╗\n", "header")
        w("║          COMPARISON REPORT — NUMERICAL SHOWDOWN       ║\n", "header")
        w("╚══════════════════════════════════════════════════════╝\n\n", "header")
        w(f"  f(x) = {self.func_var.get()}\n\n", "sub")

        # Per-method results
        w("  PERFORMANCE BENCHMARKS\n", "sub")
        w("  " + "─" * 52 + "\n", "muted")

        min_iters = None
        for m, (root, iters, _, _) in self.results.items():
            if root is not None and (min_iters is None or iters < min_iters):
                min_iters = iters

        for method, (root, iters, history, ms) in self.results.items():
            w(f"\n  {method}\n", method)
            if root is None:
                w(f"    ✗ Failed to converge\n", "fail")
            else:
                winner = iters == min_iters
                tag = "win" if winner else "normal"
                w(f"    Root      : {root:.10f}\n", tag)
                w(f"    Iters     : {iters}", tag)
                if winner:
                    w("  ← fastest\n", "win")
                else:
                    w("\n")
                w(f"    Time      : {ms:.4f} ms\n", "muted")
                if history:
                    w(f"    Final err : {history[-1]:.2e}\n", "muted")

        # Recommendation engine
        w("\n\n  RECOMMENDATION ENGINE\n", "sub")
        w("  " + "─" * 52 + "\n", "muted")

        successful = {m: v for m, v in self.results.items() if v[0] is not None}
        failed     = [m for m, v in self.results.items() if v[0] is None]

        if not successful:
            w("\n  All methods failed to converge on this function.\n"
              "  Try a different interval or initial guess.\n", "fail")
        else:
            # Check smoothness (heuristic via derivative variation)
            xs_check = np.linspace(-2, 2, 50)
            diffs = []
            for xi in xs_check:
                try:
                    diffs.append(abs((f(xi + 1e-5) - f(xi - 1e-5)) / 2e-5))
                except Exception:
                    pass
            smooth = (np.std(diffs) < 5) if diffs else True

            newton_ok = "Newton" in successful
            bisect_ok = "Bisection" in successful
            secant_ok = "Secant"   in successful

            if smooth and newton_ok:
                best = "Newton"
                reason = ("f(x) appears smooth and differentiable — Newton's quadratic\n"
                          "    convergence makes it the optimal choice here.")
            elif secant_ok:
                best = "Secant"
                reason = ("f(x) may not be ideally smooth — Secant avoids derivative\n"
                          "    computation while achieving superlinear convergence.")
            elif bisect_ok:
                best = "Bisection"
                reason = ("Bisection is recommended as the safest fallback: guaranteed\n"
                          "    linear convergence regardless of function properties.")
            else:
                best = min(successful, key=lambda m: successful[m][1])
                reason = "Chosen by fewest iterations among converging methods."

            w(f"\n  Recommended → ", "normal")
            w(f"{best}\n", best)
            w(f"    {reason}\n\n", "normal")

        if failed:
            w("  Failed methods:\n", "warn")
            for m in failed:
                w(f"    ✗ {m} — check interval / initial guess / α value\n", "fail")

        w("\n\n  METHOD SELECTION GUIDE\n", "sub")
        w("  " + "─" * 52 + "\n", "muted")
        guide = [
            ("Bisection",   "Guaranteed convergence; needs bracket [a,b]; slowest"),
            ("Newton",      "Quadratic speed; needs smooth f(x); can fail near flat slope"),
            ("Secant",      "Superlinear; no derivative required; needs 2 initial pts"),
            ("Fixed-Point", "Depends on contraction mapping; sensitive to α choice"),
        ]
        for m, desc in guide:
            w(f"\n  {m:<14}", m)
            w(f"{desc}\n", "muted")

        w("\n\n", "muted")
        rt.config(state="disabled")
        self._show_tab("Report")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = NumericalShowdown()
    app.mainloop()