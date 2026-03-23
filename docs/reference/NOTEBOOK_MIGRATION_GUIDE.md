# Guía de Migración de Notebooks para COSMIC v0.0.1

## 🎯 Objetivo

Esta guía te ayuda a migrar notebooks existentes de COSMIC a la nueva estructura modular v0.0.1.

## 🔄 Cambios Principales

### 1. Sistema de Imports

#### ❌ Antiguo (funciona pero no recomendado)
```python
import sys
sys.path.append('/Users/usuario/ruta/absoluta/COSMIC/')
from COSMIC import ClusterAnalyzer
import pytensor.tensor as pt  # Dependencias específicas
```

#### ✅ Nuevo (recomendado)
```python
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[1]  # Ajustar según ubicación

# Imports modulares específicos
from cosmic.analysis.analyzer import ClusterAnalyzer
from cosmic.core.clustering import Clustering
from cosmic.io.loader import DataLoader
from cosmic.preprocess.preprocessor import DataPreprocessor
```

### 2. Compatibilidad Legacy

Los imports antiguos **siguen funcionando** gracias a los shims:

```python
# Estos imports antiguos funcionan igual
from clustering import HDBSCANClustering  # → cosmic.core.clustering.Clustering
from data_loader import DataLoader        # → cosmic.io.loader.DataLoader
from data_preprocessor import DataPreprocessor  # → cosmic.preprocess.preprocessor.DataPreprocessor
from cluster_analysis import ClusterAnalyzer    # → cosmic.analysis.analyzer.ClusterAnalyzer
```

## 📝 Pasos de Migración

### Paso 1: Actualizar Configuración Inicial

#### Antes:
```python
import os
import sys
sys.path.append('/ruta/absoluta/al/proyecto/')
```

#### Después:
```python
import os
import sys
from pathlib import Path

# Configuración automática relativa
PROJECT_ROOT = Path.cwd().parents[1]  # Ajustar según ubicación del notebook
DATA_ROOT = PROJECT_ROOT / 'data'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verificar instalación
try:
    import cosmic
    print("✅ COSMIC v0.0.1 disponible")
except ImportError:
    print("Instalando COSMIC...")
    os.system(f"pip install -e {PROJECT_ROOT}")
```

### Paso 2: Modernizar Imports

#### Opción A: Migración Gradual (recomendado)
```python
# Mantener imports antiguos funcionando
from clustering import HDBSCANClustering as Clustering
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from cluster_analysis import ClusterAnalyzer

# Gradualmente cambiar a:
# from cosmic.core.clustering import Clustering
# from cosmic.io.loader import DataLoader
# etc.
```

#### Opción B: Migración Completa
```python
# Imports modernos directamente
from cosmic.analysis.analyzer import ClusterAnalyzer
from cosmic.core.clustering import Clustering
from cosmic.io.loader import DataLoader
from cosmic.preprocess.preprocessor import DataPreprocessor
```

### Paso 3: Verificar Funcionalidad

```python
# Test básico para verificar que todo funciona
print("🧪 Verificando módulos...")
print(f"✅ ClusterAnalyzer: {ClusterAnalyzer}")
print(f"✅ Clustering: {Clustering}")  
print(f"✅ DataLoader: {DataLoader}")
print(f"✅ DataPreprocessor: {DataPreprocessor}")
```

## 🔧 Casos Específicos

### Notebooks en Subdirectorios

Si tu notebook está en `examples/proyecto/mi_notebook.ipynb`:

```python
# Ajustar PROJECT_ROOT según la profundidad
PROJECT_ROOT = Path.cwd().parents[2]  # Dos niveles arriba
```

### Notebooks con Dependencias Específicas

```python
# Antes
import pytensor.tensor as pt
from astropy.coordinates import SkyCoord

# Después (agregar verificaciones)
try:
    import pytensor.tensor as pt
    PYTENSOR_AVAILABLE = True
except ImportError:
    PYTENSOR_AVAILABLE = False
    print("⚠️ PyTensor no disponible, algunas funciones limitadas")

# Imports de COSMIC continúan normal
from cosmic.analysis.analyzer import ClusterAnalyzer
```

### Notebooks con Rutas de Datos Hardcoded

#### Antes:
```python
data_file = "/Users/usuario/datos/NGC6383/cluster_data.ecsv"
```

#### Después:
```python
# Rutas relativas automáticas
DATA_ROOT = PROJECT_ROOT / 'data'
data_file = DATA_ROOT / "test" / "NGC6383" / "cluster_data.ecsv"

# Verificación
if not data_file.exists():
    print(f"⚠️ Archivo no encontrado: {data_file}")
    # Buscar alternativas o crear datos de ejemplo
```

## 📊 Ejemplo Completo de Migración

### Notebook Antiguo:
```python
# Celda 1
import sys
sys.path.append('/Users/usuario/COSMIC/')
from COSMIC import ClusterAnalyzer

# Celda 2  
analyzer = ClusterAnalyzer('/Users/usuario/datos/cluster.csv')
results = analyzer.run_analysis()
```

### Notebook Migrado:
```python
# Celda 1: Configuración
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[1]
DATA_ROOT = PROJECT_ROOT / 'data'

# Imports modernos
from cosmic.analysis.analyzer import ClusterAnalyzer

# Celda 2: Análisis
data_file = DATA_ROOT / 'cluster.csv'
analyzer = ClusterAnalyzer(data=str(data_file))
results = analyzer.run_analysis()
```

## ✅ Checklist de Migración

- [ ] Actualizar configuración de rutas (relativas vs absolutas)
- [ ] Modernizar imports (opcional pero recomendado)  
- [ ] Verificar que los módulos se importan correctamente
- [ ] Actualizar rutas de archivos de datos
- [ ] Probar funcionalidad básica
- [ ] Documentar cambios específicos del proyecto

## 🆘 Resolución de Problemas

### Error: "No module named 'cosmic'"
```bash
# Instalar COSMIC en modo desarrollo
cd /ruta/al/proyecto/COSMIC
pip install -e .
```

### Error: "File not found"
```python
# Verificar rutas
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"Archivos en DATA_ROOT: {list(DATA_ROOT.iterdir()) if DATA_ROOT.exists() else 'No existe'}")
```

### Imports Legacy No Funcionan
```python
# Verificar que los shims están presentes
import os
shims = ['clustering.py', 'data_loader.py', 'data_preprocessor.py', 'cluster_analysis.py']
for shim in shims:
    shim_path = PROJECT_ROOT / shim
    print(f"{shim}: {'✅' if shim_path.exists() else '❌'}")
```

## 🚀 Beneficios de la Migración

1. **Modularidad**: Imports más específicos y claros
2. **Portabilidad**: Rutas relativas automáticas
3. **Mantenibilidad**: Estructura organizada
4. **Compatibilidad**: Los notebooks antiguos siguen funcionando
5. **Futuro**: Preparado para nuevas funcionalidades

## 📚 Recursos Adicionales

- **Notebook de ejemplo**: `examples/preprocessing_demo.ipynb`
- **Documentación**: `docs/README.md`  
- **Tests**: `tests/` - para ver casos de uso
- **Ejemplos actualizados**: `examples/ngc6383/TEST.ipynb`

---

**¡La migración es opcional pero recomendada para aprovechar todas las nuevas funcionalidades!** 🌟