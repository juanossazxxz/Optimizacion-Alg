# Docs-Optimizacion
Documentación  algoritmos de optimización
Este repositorio contiene implementaciones y documentación de varios algoritmos de optimización, incluyendo el Método de Newton, Programación Lineal, el Método QMP (Quadratic Mean Penalty), y el Algoritmo Heurístico de Firefly.

## Contenido

1. [Método de Newton](#método-de-newton)
2. [Programación Lineal](#programación-lineal)
3. [Método QMP (Quadratic Mean Penalty)](#método-qmp-quadratic-mean-penalty)
4. [Algoritmo Heurístico de Firefly](#algoritmo-heurístico-de-firefly)

---

## Método de Newton

### Descripción
El Método de Newton es un algoritmo de optimización de segundo orden utilizado para encontrar máximos o mínimos de una función. Utiliza tanto el gradiente como la matriz Hessiana de la función para realizar actualizaciones más precisas de los parámetros.

### Implementación

```python
import numpy as np

def metodo_de_newton(f, grad_f, hess_f, x0, n_iter=10):
    x = x0
    for _ in range(n_iter):
        gradiente = grad_f(x)
        hessiana = hess_f(x)
        x = x - np.linalg.inv(hessiana).dot(gradiente)
    return x
