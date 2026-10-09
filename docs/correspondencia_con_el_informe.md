# Correspondencia con el Informe Técnico N.° 1

Cada resultado del informe, el código que lo produce y el archivo donde queda.
Las rutas de salida son relativas a la carpeta `salida` de `configuracion/rutas.json`.

## Metodología

| Informe | Contenido | Código |
| --- | --- | --- |
| Sección 2.1, Tabla 2 | Muestra y partición por niño | manifiestos de entrada; verificado en `07_verificar_cifras.py` |
| Sección 2.3 | Validez, interpolación, centrado y escala | `diana_comparativo/acondicionamiento.py` |
| Sección 2.4, Tabla 3 | Costo local, soporte mínimo 9/17, DTW, abstención | `diana_comparativo/dtw.py` |
| Sección 2.5 | Referencia primaria y conjunto top-k | `diana_comparativo/referencias.py` |
| Sección 2.5 | Plantillas adaptativas (p-medianas) | `diana_comparativo/estabilidad.py` (`solve_greedy_swap`, `refine_swaps`) |
| Sección 2.6 | Remuestreo, criterios de estabilidad y de soporte | `diana_comparativo/estabilidad.py` (`build_resampling`, `set_geometry`, `assignment_comparison`, `select_k_by_group`, `select_n_by_group`) |

## Resultados

| Informe | Resultado | Script | Archivo |
| --- | --- | --- | --- |
| Sección 2.3 | 1,0 % de faltantes tras interpolar; 326 ejecuciones con origen alternativo | `01_acondicionar_secuencias.py` | `resumenes/acondicionamiento_por_ejecucion.csv` |
| Sección 3.1 | Reproducción exacta de las 4.070 secuencias | `01_acondicionar_secuencias.py --comparar-con <carpeta>` | salida en consola |
| Figura 2 | Soporte por acción y grupo etario | `06_generar_figuras.py` | `figuras/Fig2_soporte_entrenamiento.png` |
| Tabla 4 | 52 primarias y 194 top-k | `03_seleccionar_referencias.py` | `referencias/catalogo_primarias.csv`, `referencias/catalogo_topk.csv` |
| Tabla 4, Figura 3 | 85 plantillas adaptativas (k = 1, 2, 3) | `04_estudio_estabilidad.py` | `estabilidad/<id>/tables/final_template_catalog.csv`, `k_selection_by_group.csv` |
| Sección 3.2 | Reducción de 5,4 % frente a una plantilla | `04_estudio_estabilidad.py` | `estabilidad/<id>/tables/policy_comparison.csv` |
| Sección 3.2 | Revisión visual: 5 confirmadas, 1 descartada, 46 pendientes | `resumen_revision_visual.py` | salida en consola |
| Tabla 5, Figura 4 | Estabilidad por grupo y según n | `04_estudio_estabilidad.py` | `estabilidad/<id>/tables/group_status_summary.csv`, `sample_size_curves.csv` |
| Tabla 6 | Matrices de distancias por grupo | `02_matrices_por_grupo.py` | `matrices/` |
| Tabla 6, Sección 3.4 | 12.860 comparaciones; 10.716 distancias; 2.144 abstenciones | `05_comparar_con_referencias.py` | `comparacion/comparaciones.csv`, `comparacion/solicitudes.csv` |
| Tabla 6 | Resultados agregados por ejecución y por niño-acción | `05_comparar_con_referencias.py` | `comparacion/perfiles_por_ejecucion.csv`, `comparacion/agregados_nino_accion.csv` |
| Figura 5 | Cobertura por acción | `06_generar_figuras.py` | `figuras/Fig5_cobertura_por_accion.png` |
| Tabla 7 | Cobertura y variabilidad por resumen | `05_comparar_con_referencias.py` | `comparacion/cobertura_por_resumen.csv` |
| Sección 4.3 | Controles de separación y completitud | `05_comparar_con_referencias.py` | `comparacion/controles.json` |

`<id>` es el identificador del estudio, por defecto `estudio_200_semilla20260715`.

## Columnas principales de `comparacion/comparaciones.csv`

| Columna | Significado |
| --- | --- |
| `id_ejecucion`, `id_nino` | ejecución de validación y niño |
| `accion`, `grupo_etario` | acción y grupo etario de la ejecución |
| `desplazamiento_etario` | -1, 0 o +1 grupo de 12 meses |
| `id_referencia`, `rango_topk`, `es_primaria` | referencia comparada |
| `distancia` | distancia DTW (vacía si hay abstención) |
| `estado` | `calculada` o `sin_camino_admisible` |
| `version_dtw`, `version_acondicionamiento`, `soporte_minimo` | metadatos del procedimiento |
