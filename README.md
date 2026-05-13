# The Numerical Showdown 🧮

An interactive, high-performance benchmarking suite for root-finding algorithms. This tool allows users to visualize and compare the convergence speeds, computational costs, and stability of different numerical methods in real-time.

## Features
* **Custom GUI Framework:** A sleek, dark-mode dashboard built from scratch over standard Tkinter, featuring hover states, dynamic color coding, and tabbed data visualization.
* **No-Library Math Engine:** Core solvers (Bisection, Newton, Secant, Fixed-Point) are implemented from first principles, ensuring complete transparency of the underlying algorithmic logic.
* **Live Benchmarking:** Tracks iteration counts and microsecond execution times (`time.perf_counter()`) to compare true computational efficiency.
* **Algorithmic Consultant:** A built-in heuristic engine that analyzes function smoothness and convergence behavior to recommend the optimal numerical method.

## The Contenders
1. **Bisection:** The ultimate fallback. Guaranteed linear convergence.
2. **Newton-Raphson:** The quadratic sprinter. Requires a differentiable function.
3. **Secant Method:** Super-linear convergence without the need for manual derivatives.
4. **Fixed-Point Iteration:** Transformative approach sensitive to contraction mapping parameters.

## Installation & Usage
Ensure you have Python 3.8+ installed, then run:

```bash
pip install numpy matplotlib
python interface.py