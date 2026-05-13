import time
from math_utils import get_derivative

def bisection(f, a, b, eps, max_iter=100):
    if f(a) * f(b) >= 0:
        return None, 0, [], []
    
    history = []
    t_history = []
    t0 = time.perf_counter()
    iters = 0
    
    while (b - a) / 2 > eps and iters < max_iter:
        iters += 1
        c = (a + b) / 2
        error = (b - a) / 2
        
        history.append(error)
        t_history.append((time.perf_counter() - t0) * 1000)
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
            
    return c, iters, history, t_history

def newton(f, x0, eps, max_iter=100):
    x = x0
    history = []
    t_history = []
    t0 = time.perf_counter()
    
    for i in range(max_iter):
        fx = f(x)
        dfx = get_derivative(f, x)
        
        if abs(dfx) < 1e-12: 
            break
            
        x_next = x - fx / dfx
        error = abs(x_next - x)
        
        history.append(error)
        t_history.append((time.perf_counter() - t0) * 1000)
        
        if error < eps:
            return x_next, i + 1, history, t_history
        x = x_next
        
    return x, max_iter, history, t_history

def secant(f, x0, x1, eps, max_iter=100):
    history = []
    t_history = []
    t0 = time.perf_counter()
    
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        
        if abs(fx1 - fx0) < 1e-12:
            break
            
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x2 - x1)
        
        history.append(error)
        t_history.append((time.perf_counter() - t0) * 1000)
        
        if error < eps:
            return x2, i + 1, history, t_history
        x0, x1 = x1, x2
        
    return x1, max_iter, history, t_history

def fixed_point(f, x0, alpha, eps, max_iter=100):
    x = x0
    history = []
    t_history = []
    t0 = time.perf_counter()
    
    for i in range(max_iter):
        x_next = x + alpha * f(x)
        error = abs(x_next - x)
        
        history.append(error)
        t_history.append((time.perf_counter() - t0) * 1000)
        
        if error < eps:
            return x_next, i + 1, history, t_history
        x = x_next
        
    return x, max_iter, history, t_history