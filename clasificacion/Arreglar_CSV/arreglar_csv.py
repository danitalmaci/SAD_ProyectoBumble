import csv
import re

entrada = "Bumble.csv"
salida = "Bumble_limpio.csv"

patron_fecha = re.compile(r"^\d{4}-\d{2}-\d{2}$")
generos_validos = {"male", "female"}
scores_validos = {"1", "2", "3", "4", "5"}

filas_ok = 0
filas_mal = 0

with open(entrada, "r", encoding="utf-8", errors="replace") as f_in, \
     open(salida, "w", encoding="utf-8", newline="") as f_out:

    writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
    writer.writerow(["reviewId", "content", "score", "gender", "location", "date"])

    next(f_in, None)  # saltar cabecera original

    for num_linea, linea in enumerate(f_in, start=2):
        linea = linea.strip()

        if not linea:
            continue

        # quitar coma final sobrante si existe
        if linea.endswith(","):
            linea = linea[:-1]

        partes = [p.strip() for p in linea.split(",")]

        try:
            review_id = partes[0]

            # buscar la fecha desde la derecha
            idx_fecha = None
            for i in range(len(partes) - 1, 0, -1):
                if patron_fecha.match(partes[i].strip().strip('"')):
                    idx_fecha = i
                    break

            if idx_fecha is None:
                raise ValueError("No se encontró fecha")

            date = partes[idx_fecha].strip().strip('"')

            # buscar género antes de la fecha
            idx_genero = None
            for i in range(idx_fecha - 1, 0, -1):
                val = partes[i].strip().strip('"').lower()
                if val in generos_validos:
                    idx_genero = i
                    break

            if idx_genero is None:
                raise ValueError("No se encontró gender")

            gender = partes[idx_genero].strip().strip('"').lower()

            # buscar score antes del gender
            idx_score = None
            for i in range(idx_genero - 1, 0, -1):
                val = partes[i].strip().strip('"')
                if val in scores_validos:
                    idx_score = i
                    break

            if idx_score is None:
                raise ValueError("No se encontró score")

            score = partes[idx_score].strip().strip('"')

            # reconstruir content y location
            content = ",".join(partes[1:idx_score]).strip().strip('"')
            location = ",".join(partes[idx_genero + 1:idx_fecha]).strip().strip('"')

            writer.writerow([review_id, content, score, gender, location, date])
            filas_ok += 1

        except Exception as e:
            print(f"Línea {num_linea} mal reconstruida: {e}")
            filas_mal += 1

print(f"Filas bien: {filas_ok}")
print(f"Filas mal: {filas_mal}")
print(f"Archivo creado: {salida}")
