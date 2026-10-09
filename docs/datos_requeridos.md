# Datos requeridos

Los insumos residen en el almacenamiento local autorizado del proyecto y se
ubican mediante `configuracion/rutas.json`. Ninguno se versiona.

## Manifiestos (CSV)

| Archivo (`rutas.json`) | Contenido | Columnas utilizadas |
| --- | --- | --- |
| `manifiesto_elegibles` | 4.070 ejecuciones elegibles | `segment_id`, `excel_children`, `action_code`, `pose2d_npz_path` |
| `manifiesto_entrenamiento` | 2.809 ejecuciones de 109 niños | `segment_id`, `excel_children`, `action_code`, `age_bin_12m_label` |
| `manifiesto_validacion` | 1.261 ejecuciones de 48 niños | las mismas |

- `segment_id`: identificador de la ejecución (una repetición segmentada).
- `excel_children`: identificador seudónimo del niño; define la partición.
- `age_bin_12m_label`: grupo etario (`36_47mo`, `48_59mo`, `60_71mo`, `72_83mo`).

## Pose 2D (NPZ, una por ejecución)

`keypoints` (T × 17 × 2 o más columnas; se usan x e y), `confidence` (T × 17)
y, opcionalmente, `metadata_json`. El valor (0, 0) indica una articulación no
detectada.

## Secuencias normalizadas (NPZ, salida del paso 1)

| Arreglo | Forma | Contenido |
| --- | --- | --- |
| `sequence` | T × 17 × 2 | coordenadas centradas y escaladas; NaN si no son válidas |
| `valid_mask` | T × 17 | validez, incluidas las interpolaciones permitidas |
| `confidence` | T × 17 | confianza original |
| `keypoint_names` | 17 | orden COCO17 |
| `metadata_json` | texto | versión y parámetros del acondicionamiento, escala y diagnósticos |

## Matrices de distancias por grupo

Índice CSV con `action_code`, `age_bin_12m_label`, `part`, `n_clips`,
`n_pairs` y `pairwise_path`; cada matriz es un CSV largo con `segment_id_a`,
`segment_id_b` y `distance`.

## Catálogos de referencias congelados

- Primarias: `action_code`, `age_bin_12m_label`, `medoid_segment_id`.
- Top-k: `action_code`, `age_bin_12m_label`, `segment_id`, `rank`, `excel_children`.

## Registro de revisión visual

JSON por línea, solo de anexión: `reviewer_id`, `reference_id`, `revision` y
`answers` (`final_status` PASS/REVIEW/FAIL, `actor_identity`, `failure_reasons`).
