# Catedra

## Experimento 1: comparar modelos de texto

El script de `Experimento1/Codigo` compara tres modelos para clasificar las
emociones del `spanish-meacorpus-2023-dataset.csv`. El corpus ya está preparado
para usarse: ejecuta el script desde esa carpeta y leerá el CSV automáticamente:

```powershell
python exp1_linea_base_textual.py
```

También puedes ejecutar solo uno, por ejemplo:

```powershell
python exp1_linea_base_textual.py --models A
```

Para una prueba rápida con una muestra reducida:

```powershell
python exp1_linea_base_textual.py --quick
```

El modelo B usa DistilBERT multilingüe y mantiene congelado el modelo base:
solo entrena la capa de clasificación, lo que reduce mucho el uso de memoria.
El modelo B necesita `torch` y `transformers`; el modelo C necesita
`sentence-transformers`. Si falta una dependencia, el script avisa y continúa
con los modelos disponibles. Los resultados se guardan en `resultados_exp1`.

El MEACorpus clasifica **emociones**, no el riesgo de baja de clientes. Para
estudiar bajas haría falta un CSV con etiquetas que indiquen qué clientes se
dieron de baja o estaban en riesgo.