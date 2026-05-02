import pandas as pd
import os
from sklearn.model_selection import train_test_split

def dividir_y_guardar_estratificado(ruta_csv, target_col, tamaño_dev=0.15, tamaño_test=0.15, random_state=42):
    """
    Divide el dataset estratificadamente en train, dev y test, y guarda los 3 CSV.
    
    Parámetros:
    - ruta_csv: str, ruta del archivo CSV original
    - target_col: str, nombre de la columna objetivo
    - tamaño_dev: float, proporción para validación (default 0.15)
    - tamaño_test: float, proporción para test (default 0.15)
    - random_state: int, semilla para reproducibilidad
    """
    
    # 1. Cargar datos
    print(f"📂 Cargando: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print(f"✅ Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Validar columna objetivo
    if target_col not in df.columns:
        raise ValueError(f"❌ Columna '{target_col}' no encontrada")
    
    # 2. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Mostrar distribución original
    print(f"\n📊 Distribución original de '{target_col}':")
    print(y.value_counts())
    print(f"Proporciones:\n{y.value_counts(normalize=True)}")
    
    # 3. Primera división: separar TEST
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=tamaño_test,
        stratify=y,
        random_state=random_state
    )
    
    # 4. Segunda división: separar TRAIN y DEV del resto
    tamaño_dev_ajustado = tamaño_dev / (1 - tamaño_test)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp,
        test_size=tamaño_dev_ajustado,
        stratify=y_temp,
        random_state=random_state
    )
    
    # 5. Reconstruir DataFrames completos (con la columna objetivo)
    train_df = X_train.copy()
    train_df[target_col] = y_train
    
    dev_df = X_dev.copy()
    dev_df[target_col] = y_dev
    
    test_df = X_test.copy()
    test_df[target_col] = y_test
    
    # 6. Generar nombres de archivo
    base_name = ruta_csv.rsplit('.', 1)[0]  # Quita la extensión
    train_path = f"{base_name}_train.csv"
    dev_path = f"{base_name}_dev.csv"
    test_path = f"{base_name}_test.csv"
    
    # 7. Guardar los 3 CSV
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # 8. Mostrar estadísticas
    print(f"\n{'='*50}")
    print("📁 ARCHIVOS GUARDADOS:")
    print(f"  ✓ {os.path.basename(train_path)} ({len(train_df)} filas, {len(train_df)/len(df)*100:.1f}%)")
    print(f"  ✓ {os.path.basename(dev_path)} ({len(dev_df)} filas, {len(dev_df)/len(df)*100:.1f}%)")
    print(f"  ✓ {os.path.basename(test_path)} ({len(test_df)} filas, {len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\n📊 VERIFICACIÓN DE ESTRATIFICACIÓN:")
    print(f"Train - {target_col}:")
    print(train_df[target_col].value_counts(normalize=True))
    print(f"\nDev - {target_col}:")
    print(dev_df[target_col].value_counts(normalize=True))
    print(f"\nTest - {target_col}:")
    print(test_df[target_col].value_counts(normalize=True))
    
    print(f"\n{'='*50}")
    print(f"✅ División completada. Los 3 CSV están en: {os.path.dirname(ruta_csv)}")
    
    return train_path, dev_path, test_path  # Retorna las rutas por si las necesitas


# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    print("="*50)
    print("DIVISOR ESTRATIFICADO DE DATASET (Train/Dev/Test)")
    print("="*50)
    
    # Solicitar datos al usuario
    ruta = input("\n📂 Ruta del archivo CSV: ").strip()
    columna_objetivo = input("🎯 Nombre de la columna objetivo: ").strip()
    
    # Parámetros opcionales
    print("\n⚙️  Configuración (opcional, presiona Enter para usar valores por defecto):")
    test_input = input("  Proporción para Test [0.15]: ").strip()
    dev_input = input("  Proporción para Dev/Validación [0.15]: ").strip()
    
    # Asignar valores
    test_size = float(test_input) if test_input else 0.15
    dev_size = float(dev_input) if dev_input else 0.15
    train_size = 1 - test_size - dev_size
    
    print(f"\n📐 Distribución final: Train={train_size:.1%} | Dev={dev_size:.1%} | Test={test_size:.1%}")
    
    try:
        # Ejecutar división y guardado
        train_file, dev_file, test_file = dividir_y_guardar_estratificado(
            ruta_csv=ruta,
            target_col=columna_objetivo,
            tamaño_dev=dev_size,
            tamaño_test=test_size,
            random_state=42  # Cambia para diferente aleatoriedad
        )
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo '{ruta}'")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")