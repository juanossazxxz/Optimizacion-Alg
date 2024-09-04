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
```
## Ejemplo de uso

```python
def f(x):
    return x**2

def grad_f(x):
    return 2*x

def hess_f(x):
    return np.array([[2]])

x_inicial = np.array([10.0])
minimo = metodo_de_newton(f, grad_f, hess_f, x_inicial)
print(f"El mínimo encontrado está en: {minimo}")
```
## Progrmacion lineal

### Descripción
La Programación Lineal es un método para maximizar o minimizar una función lineal sujeta a restricciones lineales. Es ampliamente utilizada en optimización de recursos y problemas de planificación.

### Implementación

```python
from scipy.optimize import linprog

def programacion_lineal(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex')
    return resultado
```
### Ejemplo de uso 

```python
c = [-3, -2]  # Coeficientes de la función objetivo (invertidos para maximización)
A_ub = [[2, 1], [-4, 5]]  # Coeficientes de las restricciones
b_ub = [20, -10]  # Lado derecho de las restricciones
bounds = [(0, None), (0, None)]  # x >= 0, y >= 0

resultado = programacion_lineal(c, A_ub, b_ub, bounds=bounds)
print(f"La solución óptima es: {resultado.x}")
print(f"Valor de la función objetivo: {-resultado.fun}")  # Revertimos el signo para obtener la maximización

```
## Método QMP (Quadratic Mean Penalty)

### Descripción
El Método QMP es una técnica de optimización que convierte un problema con restricciones en uno sin restricciones utilizando una función de penalización cuadrática.

### Implementación

```python
def metodo_qmp(f, g, x0, rho=1.0, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        penalizacion = sum(max(0, gi(x))**2 for gi in g)
        qmp_obj = lambda x: f(x) + rho * penalizacion
        
        x_new = minimizar(qmp_obj, x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def minimizar(func, x0):
    # Implementa un método de optimización como Gradiente Descendiente o Newton
    pass

```
### Ejemplo de uso 

```python
import numpy as np

# Definir la función objetivo
def f(x):
    return x[0]**2 + x[1]**2

# Definir las restricciones (g(x) <= 0)
def g1(x):
    return x[0] + x[1] - 1

def g2(x):
    return -x[0]

# Ejecutar el método QMP
x_inicial = np.array([2.0, 2.0])
solucion = metodo_qmp(f, [g1, g2], x_inicial)
print(f"La solución óptima es: {solucion}")

```

## Algoritmo Heurístico de Firefly

### Descripción
El Algoritmo de Firefly es una metaheurística basada en la naturaleza que simula el comportamiento de las luciérnagas. Las luciérnagas se mueven en el espacio de búsqueda hacia aquellas que son más brillantes, es decir, que representan soluciones de mayor calidad.

### Implementación

```python
import numpy as np

def algoritmo_firefly(f, n, alpha=0.5, beta0=1, gamma=1.0, max_iter=100):
    dim = len(f(np.zeros(n)))  # Asume que la función f devuelve un vector de longitud dim
    luciernagas = np.random.rand(n, dim)
    fitness = np.apply_along_axis(f, 1, luciernagas)
    
    for t in range(max_iter):
        for i in range(n):
            for j in range(n):
                if fitness[i] < fitness[j]:
                    r = np.linalg.norm(luciernagas[i] - luciernagas[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    luciernagas[i] = luciernagas[i] * (1 - beta) + beta * luciernagas[j] + alpha * (np.random.rand(dim) - 0.5)
                    fitness[i] = f(luciernagas[i])
                    
        mejor_solucion = luciernagas[np.argmin(fitness)]
        mejor_fitness = np.min(fitness)
        
    return mejor_solucion, mejor_fitness

```

### Ejemplo de uso 

```python
# Ejemplo de función objetivo: Esfera (minimizar)
def esfera(x):
    return np.sum(x**2)

n_luciernagas = 20
dim = 2
max_iteraciones = 100

mejor_solucion, mejor_fitness = algoritmo_firefly(esfera, n_luciernagas, max_iter=max_iteraciones)
print(f"La mejor solución encontrada es: {mejor_solucion}")
print(f"Valor de la función objetivo: {mejor_fitness}")

```

## Referencias 
- Dantzig, G. B. (1998). Linear Programming and Extensions. Princeton University Press.
- Fletcher, R. (1987). Practical Methods of Optimization. John Wiley & Sons.
- Yang, X. S. (2010). Nature-Inspired Metaheuristic Algorithms. Luniver Press.
- Bertsekas, D. P. (1999). Nonlinear Programming. Athena Scientific.











