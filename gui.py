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

# ══════════════════════════════════════════════════════
#  THEME  —  deep navy + neon coral/mint/gold palette
# ══════════════════════════════════════════════════════
BG        = "#09090f"
PANEL     = "#0f0f1a"
CARD      = "#13131f"
CARD2     = "#1a1a2e"
BORDER    = "#252540"
BORDER2   = "#3a3a60"

ACCENT    = "#a78bfa"          # soft violet  (primary)
ACCENT2   = "#34d399"          # emerald green
ACCENT3   = "#f472b6"          # pink

SUCCESS   = "#34d399"
WARNING   = "#fbbf24"
DANGER    = "#f87171"
INFO      = "#60a5fa"

TEXT      = "#f1f0ff"
TEXT_MUT  = "#6b6a8a"
TEXT_DIM  = "#3a3958"

METHOD_COLORS = {
    "Bisection":   "#a78bfa",
    "Newton":      "#34d399",
    "Secant":      "#f472b6",
    "Fixed-Point": "#fbbf24",
}

FONT_TITLE  = ("Courier New", 21, "bold")
FONT_SUB    = ("Courier New",  9)
FONT_HEAD   = ("Courier New", 11, "bold")
FONT_BODY   = ("Courier New", 10)
FONT_MONO   = ("Courier New", 10)
FONT_SMALL  = ("Courier New",  8)
FONT_BADGE  = ("Courier New",  8, "bold")

# ══════════════════════════════════════════════════════
#  PRESETS
#  Schema: (label, expr, a, b, x0, x1, eps, alpha, note)
#
#  Presentation story:
#   #1  x³−x−2         → ALL 4 converge cleanly             ✅✅✅✅
#   #2  cos(x)−x        → ALL 4 converge                     ✅✅✅✅
#   #3  eˣ−3x           → Bisection/Newton/Secant OK;
#                          Fixed-Point diverges (alpha=+0.5, |1+α·f′|>1)
#   #4  x²−2x−3         → Bisection/Newton/Secant OK;
#                          Fixed-Point diverges (alpha=+0.8)
#   #5  x³ (triple root)→ Bisection FAILS (no sign change on [0.5,2]);
#                          Newton+Secant converge slowly;
#                          Fixed-Point diverges                FAIL×2
#   #6  log(x)−1        → Newton+Secant converge to e≈2.718;
#                          Bisection FAILS (f(-1) undefined → caught);
#                          Fixed-Point diverges (alpha=+0.3)   FAIL×2
# ══════════════════════════════════════════════════════
PRESET_FUNCTIONS = [
    # label                      expr               a      b     x0     x1     eps     alpha  note
    ("① x³ − x − 2",            "x**3 - x - 2",   "-2",  "2",  "1",   "1.5", "1e-6", "-0.2", "ALL PASS"),
    ("② cos(x) − x",            "cos(x) - x",      "0",   "2",  "0.7", "1.0", "1e-6", "-0.5", "ALL PASS"),
    ("③ eˣ − 3x",               "exp(x) - 3*x",    "0",   "2",  "1.0", "1.5", "1e-6",  "0.5", "FP FAILS"),
    ("④ x² − 2x − 3",          "x**2 - 2*x - 3",  "-1",  "0",  "-0.5","-0.2","1e-6",  "0.8", "FP FAILS"),
    ("⑤ x³  (triple root)",     "x**3",             "0.5", "2",  "0.5", "0.8", "1e-6", "-0.3", "BIS FAILS"),
    ("⑥ log(x) − 1",           "log(x) - 1",       "-1",  "4",  "2.5", "3.0", "1e-6",  "0.3", "BIS+FP FAIL"),
]

PRESET_NOTES_COLOR = {
    "ALL PASS":    SUCCESS,
    "FP FAILS":    WARNING,
    "BIS FAILS":   DANGER,
    "BIS+FP FAIL": DANGER,
}

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def make_card(parent, **kw):
    return tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                    highlightthickness=1, **kw)

def label(parent, text, font=FONT_BODY, fg=TEXT, bg=None, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=bg or parent["bg"], **kw)

def entry(parent, width=18, **kw):
    e = tk.Entry(parent, font=FONT_MONO, bg=CARD2, fg=TEXT,
                 insertbackground=ACCENT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=width, **kw)
    e.bind("<FocusIn>",  lambda _: e.config(highlightbackground=ACCENT))
    e.bind("<FocusOut>", lambda _: e.config(highlightbackground=BORDER))
    return e

def accent_btn(parent, text, cmd, color=None, **kw):
    c = color or ACCENT
    b = tk.Button(parent, text=text, command=cmd,
                  font=FONT_HEAD, fg=BG, bg=c,
                  activebackground=_lighten(c), activeforeground=BG,
                  relief="flat", cursor="hand2", padx=14, pady=9, **kw)
    b.bind("<Enter>", lambda _: b.config(bg=_lighten(c)))
    b.bind("<Leave>", lambda _: b.config(bg=c))
    return b

def _lighten(hex_color, amt=28):
    r,g,b = int(hex_color[1:3],16),int(hex_color[3:5],16),int(hex_color[5:7],16)
    return f"#{min(255,r+amt):02x}{min(255,g+amt):02x}{min(255,b+amt):02x}"

def _tint(hex_color, factor=0.12, base=CARD):
    """Blend hex_color into base at 'factor' — subtle coloured fill."""
    fr,fg_c,fb = int(hex_color[1:3],16),int(hex_color[3:5],16),int(hex_color[5:7],16)
    br,bg_c,bb = int(base[1:3],16),int(base[3:5],16),int(base[5:7],16)
    return (f"#{int(br+(fr-br)*factor):02x}"
            f"{int(bg_c+(fg_c-bg_c)*factor):02x}"
            f"{int(bb+(fb-bb)*factor):02x}")

# ══════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════

class NumericalShowdown(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("The Numerical Showdown")
        self.geometry("1300x880")
        self.minsize(1100, 780)
        self.configure(bg=BG)
        self.resizable(True, True)

        self.results             = {}
        self.active_methods      = {m: tk.BooleanVar(value=True) for m in METHOD_COLORS}
        self._active_preset_btn  = None

        self._build_layout()
        self._set_defaults()

    # ── TOP-LEVEL LAYOUT ─────────────────────────────

    def _build_layout(self):
        self._build_header()
        tk.Frame(self, bg=BORDER2, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(14, 0))
        body.columnconfigure(0, weight=0, minsize=295)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_main(body)

    def _build_header(self):
        # Top accent bar
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")

        hdr = tk.Frame(self, bg=BG, pady=12)
        hdr.pack(fill="x", padx=26)

        left = tk.Frame(hdr, bg=BG)
        left.pack(side="left")
        label(left, "THE NUMERICAL SHOWDOWN", font=FONT_TITLE,
              fg=ACCENT, bg=BG).pack(side="left")
        label(left, "  root-finding benchmark suite", font=FONT_SUB,
              fg=TEXT_MUT, bg=BG).pack(side="left", pady=(9, 0))

        badge_bg = _tint(ACCENT, 0.20, BG)
        badge = tk.Frame(hdr, bg=badge_bg,
                         highlightbackground=BORDER2, highlightthickness=1)
        badge.pack(side="right", padx=4)
        label(badge, "  v2.0  ", font=FONT_BADGE,
              fg=ACCENT, bg=badge_bg).pack(padx=6, pady=3)

    # ── SIDEBAR ──────────────────────────────────────

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG, width=295)
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        side.pack_propagate(False)

        self._build_func_card(side)
        self._build_params_card(side)
        self._build_methods_card(side)

        run_f = tk.Frame(side, bg=BG)
        run_f.pack(fill="x", pady=(6, 8))
        accent_btn(run_f, "▶   RUN  SHOWDOWN", self._run,
                   color=ACCENT).pack(fill="x")

    def _build_func_card(self, parent):
        fc = make_card(parent)
        fc.pack(fill="x", pady=(0, 10))

        hdr = tk.Frame(fc, bg=CARD)
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=ACCENT, width=3).pack(side="left", fill="y")
        label(hdr, "  f(x) = 0", font=FONT_HEAD,
              fg=ACCENT, bg=CARD).pack(side="left", padx=(6, 0), pady=8)
        tk.Frame(fc, bg=BORDER, height=1).pack(fill="x")

        ef = tk.Frame(fc, bg=CARD, pady=8)
        ef.pack(fill="x", padx=12)
        self.func_var = tk.StringVar()
        entry(ef, width=28, textvariable=self.func_var).pack(fill="x")

        lf = tk.Frame(fc, bg=CARD)
        lf.pack(fill="x", padx=12)
        label(lf, "PRESETS", font=FONT_BADGE,
              fg=TEXT_DIM, bg=CARD).pack(side="left", pady=(2, 4))
        tk.Frame(fc, bg=BORDER, height=1).pack(fill="x", padx=12)

        pf = tk.Frame(fc, bg=CARD)
        pf.pack(fill="x", padx=10, pady=(4, 10))

        for disp, expr, a, b, x0, x1, eps, alpha, note in PRESET_FUNCTIONS:
            note_col = PRESET_NOTES_COLOR.get(note, TEXT_MUT)
            row = tk.Frame(pf, bg=CARD)
            row.pack(fill="x", pady=2)

            btn = tk.Button(
                row, text=disp, font=FONT_BODY,
                fg=TEXT_MUT, bg=CARD,
                activebackground=_tint(ACCENT, 0.12, CARD),
                activeforeground=TEXT, relief="flat",
                cursor="hand2", anchor="w", padx=6, pady=3,
            )
            btn.pack(side="left", fill="x", expand=True)

            bdg_bg = _tint(note_col, 0.12, CARD)
            tk.Label(row, text=note, font=FONT_BADGE,
                     fg=note_col, bg=bdg_bg,
                     padx=4, pady=1).pack(side="right", padx=(0, 2))

            def _set(e=expr, _a=a, _b=b, _x0=x0, _x1=x1,
                     _eps=eps, _alpha=alpha, _btn=btn):
                self.func_var.set(e)
                self.p_a.set(_a);     self.p_b.set(_b)
                self.p_x0.set(_x0);   self.p_x1.set(_x1)
                self.p_eps.set(_eps);  self.p_alpha.set(_alpha)
                if self._active_preset_btn:
                    self._active_preset_btn.config(
                        fg=TEXT_MUT, bg=CARD, font=FONT_BODY)
                _btn.config(fg=ACCENT,
                            bg=_tint(ACCENT, 0.10, CARD),
                            font=("Courier New", 10, "bold"))
                self._active_preset_btn = _btn

            btn.config(command=_set)

            def _on_enter(e, b=btn):
                if b is not self._active_preset_btn:
                    b.config(fg=TEXT)
            def _on_leave(e, b=btn):
                if b is not self._active_preset_btn:
                    b.config(fg=TEXT_MUT)

            btn.bind("<Enter>", _on_enter)
            btn.bind("<Leave>", _on_leave)

    def _build_params_card(self, parent):
        pc = make_card(parent)
        pc.pack(fill="x", pady=(0, 10))

        hdr = tk.Frame(pc, bg=CARD)
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=ACCENT2, width=3).pack(side="left", fill="y")
        label(hdr, "  PARAMETERS", font=FONT_HEAD,
              fg=ACCENT2, bg=CARD).pack(side="left", padx=(6,0), pady=8)
        tk.Frame(pc, bg=BORDER, height=1).pack(fill="x")

        pf = tk.Frame(pc, bg=CARD, pady=6)
        pf.pack(fill="x")

        def row(lbl_txt, default, attr):
            r = tk.Frame(pf, bg=CARD)
            r.pack(fill="x", padx=12, pady=2)
            label(r, lbl_txt, font=FONT_SMALL, fg=TEXT_MUT,
                  bg=CARD, width=18, anchor="w").pack(side="left")
            v = tk.StringVar(value=default)
            setattr(self, attr, v)
            entry(r, width=10, textvariable=v).pack(side="right")

        row("Interval  a",     "-2",    "p_a")
        row("Interval  b",     " 2",    "p_b")
        row("Initial  x₀",    " 1",    "p_x0")
        row("Initial  x₁",    " 1.5",  "p_x1")
        row("Tolerance  ε",   "1e-6",  "p_eps")
        row("α  (fixed-pt)",  "-0.2",  "p_alpha")
        row("Max iterations",  "100",   "p_maxiter")
        tk.Frame(pc, bg=CARD, height=6).pack()

    def _build_methods_card(self, parent):
        mc = make_card(parent)
        mc.pack(fill="x", pady=(0, 6))

        hdr = tk.Frame(mc, bg=CARD)
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=ACCENT3, width=3).pack(side="left", fill="y")
        label(hdr, "  METHODS", font=FONT_HEAD,
              fg=ACCENT3, bg=CARD).pack(side="left", padx=(6,0), pady=8)
        tk.Frame(mc, bg=BORDER, height=1).pack(fill="x")

        mf = tk.Frame(mc, bg=CARD, pady=6)
        mf.pack(fill="x")
        for method, var in self.active_methods.items():
            col = METHOD_COLORS[method]
            rf  = tk.Frame(mf, bg=CARD)
            rf.pack(fill="x", padx=12, pady=2)
            tk.Checkbutton(rf, variable=var, bg=CARD,
                           fg=col, activebackground=CARD,
                           selectcolor=_tint(col, 0.30, CARD),
                           relief="flat").pack(side="left")
            tk.Frame(rf, bg=col, width=10, height=10).pack(
                side="left", padx=(2, 7))
            label(rf, method, font=FONT_BODY,
                  fg=col, bg=CARD).pack(side="left")
        tk.Frame(mc, bg=CARD, height=4).pack()

    # ── MAIN PANEL ───────────────────────────────────

    def _build_main(self, parent):
        main = tk.Frame(parent, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        self._build_stat_cards(main)
        self._build_tabs(main)

    def _build_stat_cards(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        row.columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_cards = {}
        for i, method in enumerate(METHOD_COLORS):
            col  = METHOD_COLORS[method]
            tint = _tint(col, 0.08, CARD)

            # 1-px colored border frame wrapping the inner card
            outer = tk.Frame(row, bg=col, padx=1, pady=1)
            outer.grid(row=0, column=i, sticky="ew",
                       padx=(0 if i == 0 else 7, 0))

            inner = tk.Frame(outer, bg=tint, padx=10, pady=8)
            inner.pack(fill="both", expand=True)

            label(inner, method.upper(), font=FONT_BADGE,
                  fg=col, bg=tint).pack(anchor="w")
            val_lbl = label(inner, "—",
                            font=("Courier New", 15, "bold"),
                            fg=col, bg=tint)
            val_lbl.pack(anchor="w")
            sub_lbl = label(inner, "awaiting run",
                            font=FONT_SMALL, fg=TEXT_DIM, bg=tint)
            sub_lbl.pack(anchor="w")
            self.stat_cards[method] = (val_lbl, sub_lbl, inner, tint, col)

    def _build_tabs(self, parent):
        nb = tk.Frame(parent, bg=BG)
        nb.grid(row=1, column=0, sticky="nsew")
        nb.rowconfigure(1, weight=1)
        nb.columnconfigure(0, weight=1)

        self.tab_btns   = {}
        self.tab_frames = {}

        tab_bar = tk.Frame(nb, bg=BG)
        tab_bar.grid(row=0, column=0, sticky="ew")

        self.tab_content = tk.Frame(nb, bg=PANEL,
                                    highlightbackground=BORDER2,
                                    highlightthickness=1)
        self.tab_content.grid(row=1, column=0, sticky="nsew")
        self.tab_content.rowconfigure(0, weight=1)
        self.tab_content.columnconfigure(0, weight=1)

        for tname in ("Convergence Plot", "Function Plot", "Report"):
            frame = tk.Frame(self.tab_content, bg=PANEL)
            frame.grid(row=0, column=0, sticky="nsew")
            self.tab_frames[tname] = frame

            btn = tk.Button(
                tab_bar, text=f"  {tname}  ",
                font=FONT_BODY, bg=BG, fg=TEXT_MUT, relief="flat",
                cursor="hand2", padx=8, pady=8,
                activebackground=PANEL, activeforeground=ACCENT,
                command=lambda n=tname: self._show_tab(n),
            )
            btn.pack(side="left")
            self.tab_btns[tname] = btn

        # Right-side hint
        tk.Frame(tab_bar, bg=BG).pack(side="left", fill="x", expand=True)
        label(tab_bar, "press ▶ to run  ", font=FONT_SMALL,
              fg=TEXT_DIM, bg=BG).pack(side="right", pady=8)

        self._build_convergence_tab(self.tab_frames["Convergence Plot"])
        self._build_function_tab(self.tab_frames["Function Plot"])
        self._build_report_tab(self.tab_frames["Report"])
        self._show_tab("Convergence Plot")

    def _show_tab(self, name):
        for n, btn in self.tab_btns.items():
            btn.config(fg=TEXT_MUT, bg=BG, font=FONT_BODY)
        self.tab_frames[name].lift()
        self.tab_btns[name].config(fg=ACCENT, bg=PANEL,
                                   font=("Courier New", 10, "bold"))

    # ── FIGURE SETUP ─────────────────────────────────

    def _make_figure(self):
        return plt.Figure(facecolor=PANEL, edgecolor=PANEL)

    def _style_ax(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT_MUT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER2)
            spine.set_linewidth(0.8)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(10)
        ax.title.set_fontfamily("Courier New")
        ax.title.set_fontweight("bold")
        if title:  ax.set_title(title, pad=8)
        if xlabel: ax.set_xlabel(xlabel, color=TEXT_MUT, fontsize=8,
                                 fontfamily="Courier New", labelpad=6)
        if ylabel: ax.set_ylabel(ylabel, color=TEXT_MUT, fontsize=8,
                                 fontfamily="Courier New", labelpad=6)
        ax.grid(True, color=BORDER, linewidth=0.4, linestyle="--", alpha=0.8)

    def _embed_fig(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)
        return canvas

    def _build_convergence_tab(self, frame):
        self.conv_fig = self._make_figure()
        self.conv_fig.subplots_adjust(wspace=0.32, bottom=0.14,
                                      top=0.91, left=0.08, right=0.97)
        self.ax_iter = self.conv_fig.add_subplot(121)
        self.ax_time = self.conv_fig.add_subplot(122)
        self._style_ax(self.ax_iter, title="Error vs Iterations",
                       xlabel="Iteration #", ylabel="log₁₀ (error)")
        self._style_ax(self.ax_time, title="Error vs Time (ms)",
                       xlabel="Time (ms)", ylabel="")
        self.conv_canvas = self._embed_fig(self.conv_fig, frame)

    def _build_function_tab(self, frame):
        self.func_fig = self._make_figure()
        self.func_fig.subplots_adjust(bottom=0.12, top=0.91,
                                      left=0.08, right=0.97)
        self.func_ax = self.func_fig.add_subplot(111)
        self._style_ax(self.func_ax, title="f(x) with found roots",
                       xlabel="x", ylabel="f(x)")
        self.func_canvas = self._embed_fig(self.func_fig, frame)

    def _build_report_tab(self, frame):
        wrap = tk.Frame(frame, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=14, pady=14)

        self.report_text = tk.Text(
            wrap, font=FONT_MONO, bg=CARD, fg=TEXT,
            insertbackground=ACCENT, relief="flat",
            highlightbackground=BORDER2, highlightthickness=1,
            wrap="word", state="disabled", padx=16, pady=12,
            spacing1=1, spacing3=1,
        )
        sb = ttk.Scrollbar(wrap, command=self.report_text.yview)
        self.report_text.config(yscrollcommand=sb.set)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Vertical.TScrollbar",
                        background=CARD2, troughcolor=CARD,
                        bordercolor=BORDER, arrowcolor=TEXT_MUT)

        sb.pack(side="right", fill="y")
        self.report_text.pack(fill="both", expand=True)

        tags = {
            "header": (ACCENT,   ("Courier New", 11, "bold")),
            "sub":    (ACCENT2,  ("Courier New", 10, "bold")),
            "win":    (SUCCESS,  FONT_MONO),
            "warn":   (WARNING,  FONT_MONO),
            "fail":   (DANGER,   FONT_MONO),
            "muted":  (TEXT_MUT, FONT_SMALL),
            "normal": (TEXT,     FONT_MONO),
            "info":   (INFO,     FONT_SMALL),
        }
        for tag, (fg, fnt) in tags.items():
            self.report_text.tag_config(tag, foreground=fg, font=fnt)
        for method, col in METHOD_COLORS.items():
            self.report_text.tag_config(
                method, foreground=col,
                font=("Courier New", 10, "bold"))

    # ── DEFAULTS ─────────────────────────────────────

    def _set_defaults(self):
        self.func_var.set("x**3 - x - 2")

    # ══════════════════════════════════════════════════
    #  RUN ENGINE
    # ══════════════════════════════════════════════════

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
                root, iters, history, t_history = run()
                if root is None:
                    root, iters, history, t_history = None, 0, [], []
            except Exception:
                root, iters, history, t_history = None, 0, [], []
            ms = (time.perf_counter() - t0) * 1000
            self.results[method] = (root, iters, history, t_history, ms)

        self._update_stat_cards()
        self._draw_convergence()
        self._draw_function(f, a, b)
        self._write_report(f)

    # ── UPDATE STAT CARDS ────────────────────────────

    def _update_stat_cards(self):
        min_iters = None
        for m, (root, iters, *_) in self.results.items():
            if root is not None and (min_iters is None or iters < min_iters):
                min_iters = iters

        for method, (val_lbl, sub_lbl, inner, tint, col) in self.stat_cards.items():
            if method not in self.results:
                val_lbl.config(text="—", fg=col)
                sub_lbl.config(text="not run", fg=TEXT_DIM)
                for w in (inner, val_lbl, sub_lbl):
                    w.config(bg=tint)
                continue

            root, iters, _, _, ms = self.results[method]
            if root is None:
                fail_tint = _tint(DANGER, 0.10, CARD)
                val_lbl.config(text="FAIL", fg=DANGER)
                sub_lbl.config(text="did not converge", fg=DANGER)
                for w in (inner, val_lbl, sub_lbl):
                    w.config(bg=fail_tint)
            else:
                winner = iters == min_iters
                ok_tint = _tint(col, 0.10, CARD)
                val_lbl.config(text=f"{root:.6f}", fg=col)
                badge = f"{iters} iters · {ms:.2f}ms"
                if winner:
                    badge += "  ★"
                sub_lbl.config(text=badge,
                               fg=SUCCESS if winner else TEXT_DIM)
                for w in (inner, val_lbl, sub_lbl):
                    w.config(bg=ok_tint)

    # ── CONVERGENCE PLOT ─────────────────────────────

    def _draw_convergence(self):
        self.ax_iter.clear()
        self.ax_time.clear()
        self._style_ax(self.ax_iter, title="Error vs Iterations",
                       xlabel="Iteration #", ylabel="log₁₀ (error)")
        self._style_ax(self.ax_time, title="Error vs Time (ms)",
                       xlabel="Time (ms)", ylabel="")

        plotted = False
        for method, (root, iters, history, t_history, _) in self.results.items():
            if root is None or not history:
                continue
            safe = [h for h in history if h > 0]
            if not safe:
                continue
            log_err   = np.log10(safe)
            safe_time = t_history[:len(log_err)]
            col       = METHOD_COLORS[method]
            bright    = _lighten(col, 40)

            kw = dict(color=col, linewidth=2, alpha=0.92,
                      markerfacecolor=bright, markeredgecolor=col,
                      markeredgewidth=1, label=method)
            self.ax_iter.plot(range(1, len(log_err)+1), log_err,
                              marker="o", markersize=4, **kw)
            self.ax_time.plot(safe_time, log_err,
                              marker="s", markersize=4, **kw)
            plotted = True

        leg_kw = dict(facecolor=CARD2, edgecolor=BORDER2, labelcolor=TEXT,
                      fontsize=8, prop={"family": "Courier New"}, framealpha=0.9)
        if plotted:
            self.ax_iter.legend(**leg_kw)
            self.ax_time.legend(**leg_kw)

        self.conv_fig.canvas.draw_idle()

    # ── FUNCTION PLOT ────────────────────────────────

    def _draw_function(self, f, a, b):
        ax = self.func_ax
        ax.clear()
        self._style_ax(ax, title="f(x) with found roots",
                       xlabel="x", ylabel="f(x)")

        margin = max((b - a) * 0.35, 0.5)
        xs = np.linspace(a - margin, b + margin, 700)
        try:
            ys = np.array([f(x) for x in xs], dtype=float)
        except Exception:
            self.func_fig.canvas.draw_idle()
            return

        finite = ys[np.isfinite(ys)]
        if len(finite):
            lo = np.percentile(finite, 2)
            hi = np.percentile(finite, 98)
            pad = max((hi - lo) * 0.3, 0.5)
            ax.set_ylim(lo - pad, hi + pad)

        ax.plot(xs, ys, color=ACCENT2, linewidth=2, label="f(x)", zorder=2)
        ax.axhline(0, color=BORDER2, linewidth=0.9, linestyle="--", zorder=1)
        ax.axvline(a, color=TEXT_DIM, linewidth=0.6, linestyle=":", zorder=1)
        ax.axvline(b, color=TEXT_DIM, linewidth=0.6, linestyle=":", zorder=1)

        for method, (root, iters, _, _, _) in self.results.items():
            if root is None:
                continue
            try:
                fy = f(root)
            except Exception:
                continue
            col = METHOD_COLORS[method]
            # Glow halo
            ax.plot(root, fy, "o", color=_tint(col, 0.30, CARD),
                    markersize=20, zorder=3)
            ax.plot(root, fy, "o", color=col, markersize=9, zorder=4,
                    markeredgecolor=_lighten(col, 50), markeredgewidth=1.2,
                    label=f"{method} ≈ {root:.4f}")

        ax.legend(facecolor=CARD2, edgecolor=BORDER2, labelcolor=TEXT,
                  fontsize=8, prop={"family": "Courier New"}, framealpha=0.9)
        self.func_fig.canvas.draw_idle()

    # ── REPORT ───────────────────────────────────────

    def _write_report(self, f):
        rt = self.report_text
        rt.config(state="normal")
        rt.delete("1.0", "end")

        def w(text, tag="normal"):
            rt.insert("end", text, tag)

        w("╔══════════════════════════════════════════════════════╗\n", "header")
        w("║        COMPARISON REPORT  —  NUMERICAL SHOWDOWN      ║\n", "header")
        w("╚══════════════════════════════════════════════════════╝\n\n", "header")
        w(f"  f(x)  =  {self.func_var.get()}\n\n", "sub")

        w("  PERFORMANCE BENCHMARKS\n", "sub")
        w("  " + "─" * 52 + "\n", "muted")

        min_iters = None
        for m, (root, iters, *_) in self.results.items():
            if root is not None and (min_iters is None or iters < min_iters):
                min_iters = iters

        for method, (root, iters, history, _, ms) in self.results.items():
            w(f"\n  {method}\n", method)
            if root is None:
                w("    ✗  Failed to converge\n", "fail")
            else:
                winner = iters == min_iters
                tag = "win" if winner else "normal"
                w(f"    Root      :  {root:.10f}\n", tag)
                w(f"    Iters     :  {iters}", tag)
                if winner:
                    w("  ← fastest  ★\n", "win")
                else:
                    w("\n")
                w(f"    Time      :  {ms:.4f} ms\n", "muted")
                if history:
                    w(f"    Final err :  {history[-1]:.2e}\n", "muted")

        w("\n\n  RECOMMENDATION ENGINE\n", "sub")
        w("  " + "─" * 52 + "\n", "muted")

        successful = {m: v for m, v in self.results.items() if v[0] is not None}
        failed     = [m for m, v in self.results.items() if v[0] is None]

        if not successful:
            w("\n  All methods failed to converge.\n"
              "  Try a different interval or initial guess.\n", "fail")
        else:
            xs_check = np.linspace(-2, 2, 50)
            diffs = []
            for xi in xs_check:
                try:
                    diffs.append(abs((f(xi+1e-5) - f(xi-1e-5)) / 2e-5))
                except Exception:
                    pass
            smooth = (np.std(diffs) < 5) if diffs else True

            newton_ok = "Newton"    in successful
            bisect_ok = "Bisection" in successful
            secant_ok = "Secant"    in successful

            if smooth and newton_ok:
                best   = "Newton"
                reason = ("f(x) appears smooth and differentiable — Newton's quadratic\n"
                          "    convergence makes it the optimal choice here.")
            elif secant_ok:
                best   = "Secant"
                reason = ("f(x) may not be ideally smooth — Secant avoids derivative\n"
                          "    computation while achieving superlinear convergence.")
            elif bisect_ok:
                best   = "Bisection"
                reason = ("Bisection is the safest fallback: guaranteed linear\n"
                          "    convergence regardless of function properties.")
            else:
                best   = min(successful, key=lambda m: successful[m][1])
                reason = "Chosen by fewest iterations among converging methods."

            w(f"\n  Recommended  →  ", "normal")
            w(f"{best}\n", best)
            w(f"    {reason}\n\n", "normal")

        if failed:
            w("  Failed methods:\n", "warn")
            for m in failed:
                w(f"    ✗  {m}  —  check interval / initial guess / α\n", "fail")

        w("\n\n  METHOD SELECTION GUIDE\n", "sub")
        w("  " + "─" * 52 + "\n", "muted")
        guide = [
            ("Bisection",   "Guaranteed convergence; needs bracket [a,b]; O(log n)"),
            ("Newton",      "Quadratic speed; needs smooth f(x); fails near flat f′"),
            ("Secant",      "Superlinear; no derivative; needs 2 initial points"),
            ("Fixed-Point", "Sensitive to α; needs |1 + α·f′(x*)| < 1 to converge"),
        ]
        for m, desc in guide:
            w(f"\n  {m:<14}", m)
            w(f"{desc}\n", "muted")

        w("\n\n", "muted")
        rt.config(state="disabled")
        self._show_tab("Report")


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = NumericalShowdown()
    app.mainloop()