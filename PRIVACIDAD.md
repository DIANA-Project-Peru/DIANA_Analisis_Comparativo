# Privacidad

Este repositorio contiene solo código, configuraciones y documentación. Los
datos del proyecto corresponden a menores de edad y **no se versionan**:

- videos, fotogramas, poses y secuencias normalizadas;
- manifiestos con identificadores de participantes;
- matrices de distancias, catálogos y demás resultados derivados;
- visores HTML generados con datos reales (contienen coordenadas de pose);
- rutas locales del equipo y credenciales.

Las rutas a los datos autorizados se indican en `configuracion/rutas.json`,
que está excluido de Git. Todas las salidas se escriben en la carpeta
`salida` de ese archivo, que debe estar fuera del repositorio.

Las únicas figuras incluidas en `docs/figuras/` son las del informe técnico:
contienen conteos agregados y resúmenes por grupo, sin datos individuales.
