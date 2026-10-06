# Detección de Fraude con Tarjetas de Crédito - MLOps

Este repositorio tiene el código para pasar de un entorno de experimentación manual (Notebook) hacia un pipeline de Machine Learning reproducible, modular y que deje evidencia de los logs. 

## Estructura del Repositorio

```text
├── creditcard.csv                 # Dataset original
├── requirements.txt               # Dependencias que tenemos que instalar
├── run_experiments.sh             # Script automatizado para correr varias versiones del modelo con diferentes parámetros
├── train.py                       # Script principal de entrenamiento y tracking
```

## Argumentos del Modelo (`train.py`)
El script de entrenamiento acepta los siguientes hiperparámetros:

* `--data_path`: Ruta al dataset CSV (`creditcard.csv`).
* `--c_param`: Inverso de la fuerza de regularización (Por defecto: `1.0`).
* `--class_weight`: Estrategia para balanceo de clases. Usar `None` o `balanced` (Por defecto: `None`).
* `--max_iter`: Número máximo de iteraciones permitidas para la convergencia (Por defecto: `1000`).
* `--random_state`: Semilla de para garantizar que lo reproducimos (Por defecto: `42`).

---

## Pasos para ejecutarlo

### 1. Preparación de los Datos
Descarga el dataset y coloca `creditcard.csv` en la raíz del proyecto.

### 2. Configuración del Entorno Virtual
Abre la terminal en la raíz del repositorio y ejecuta:

```bash
# Crear el entorno virtual
python3 -m venv .venv

# Activar el entorno virtual (Linux/macOS)
source .venv/bin/activate
```

### 3. Instalación de Dependencias
Con el entorno activado, instala las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt
```

### 4. Ejecución de Experimentos
Ejecuta los 5 experimentos automatizados mediante el script:

```bash
bash run_experiments.sh
```
