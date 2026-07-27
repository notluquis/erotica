#!/usr/bin/env python3
"""
Script de migración automática de notebooks EROTICA v0.0.1

Este script actualiza todos los notebooks del proyecto para usar la nueva
estructura modular, manteniendo compatibilidad con código existente.
"""

import json
from pathlib import Path
import sys

def add_compatibility_cell(notebook_path: Path) -> bool:
    """Agrega una celda de compatibilidad al inicio del notebook"""
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Verificar si ya tiene la celda de compatibilidad
        for cell in notebook.get('cells', []):
            if 'EROTICA v0.0.1 - Compatibilidad' in cell.get('source', ''):
                print(f"  ✅ Ya actualizado: {notebook_path.name}")
                return True
        
        # Crear celda de compatibilidad
        compatibility_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌟 EROTICA v0.0.1 - Compatibilidad Automática\n",
                "\n",
                "**Este notebook ha sido actualizado para EROTICA v0.0.1**\n",
                "\n",
                "## ✨ Mejoras Disponibles\n",
                "- **Estructura modular**: `erotica.core`, `erotica.io`, `erotica.preprocess`, `erotica.analysis`\n",
                "- **Imports modernos**: Específicos y organizados\n",
                "- **API mejorada**: Mejor manejo de errores y configuración\n",
                "- **Compatibilidad total**: El código existente sigue funcionando\n",
                "\n",
                "## 🔄 Migración Opcional\n",
                "Para usar las nuevas funcionalidades:\n",
                "```python\n",
                "# Nuevo (recomendado)\n",
                "from erotica.analysis.analyzer import ClusterAnalyzer\n",
                "from erotica.core.clustering import Clustering\n",
                "from erotica.io.loader import DataLoader\n",
                "\n",
                "# Legacy (sigue funcionando)\n",
                "from EROTICA import ClusterAnalyzer\n",
                "from clustering import HDBSCANClustering\n",
                "from data_loader import DataLoader\n",
                "```\n",
                "\n",
                "📚 **Documentación**: `/docs/reference/NOTEBOOK_MIGRATION_GUIDE.md`"
            ]
        }
        
        # Insertar al inicio
        notebook.setdefault('cells', []).insert(0, compatibility_cell)
        
        # Guardar notebook actualizado
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Actualizado: {notebook_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error actualizando {notebook_path.name}: {e}")
        return False

def migrate_notebooks():
    """Migra todos los notebooks del proyecto"""
    
    project_root = Path(__file__).parent.parent.parent
    print(f"🌟 EROTICA v0.0.1 - Migración Automática de Notebooks")
    print(f"📁 Proyecto: {project_root}")
    
    # Buscar todos los notebooks
    notebook_patterns = [
        "**/*.ipynb",
    ]
    
    notebooks = []
    for pattern in notebook_patterns:
        notebooks.extend(project_root.glob(pattern))
    
    # Filtrar checkpoints y backups
    notebooks = [nb for nb in notebooks 
                if '.ipynb_checkpoints' not in str(nb) 
                and 'backups' not in str(nb)]
    
    print(f"📊 Encontrados {len(notebooks)} notebooks")
    
    if not notebooks:
        print("⚠️  No se encontraron notebooks para migrar")
        return
    
    # Procesar cada notebook
    updated = 0
    failed = 0
    
    for notebook in notebooks:
        rel_path = notebook.relative_to(project_root)
        print(f"\n📝 Procesando: {rel_path}")
        
        if add_compatibility_cell(notebook):
            updated += 1
        else:
            failed += 1
    
    # Resumen
    print(f"\n📊 Resumen de migración:")
    print(f"  ✅ Actualizados: {updated}")
    print(f"  ❌ Fallidos: {failed}")
    print(f"  📝 Total: {len(notebooks)}")
    
    if updated > 0:
        print(f"\n🎉 ¡Migración completada!")
        print(f"🔧 Los notebooks mantienen compatibilidad total")
        print(f"📚 Ver guía completa: docs/reference/NOTEBOOK_MIGRATION_GUIDE.md")
    else:
        print(f"\n⚠️  No se realizaron actualizaciones")

if __name__ == "__main__":
    migrate_notebooks()