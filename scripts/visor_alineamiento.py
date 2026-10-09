#!/usr/bin/env python
"""Visor de alineamiento DTW entre una ejecución y una referencia.

Genera un archivo HTML autocontenido (sin servidor ni dependencias externas)
que muestra:

* los dos esqueletos normalizados recorriendo el camino de alineamiento DTW,
  con control deslizante y reproducción;
* la matriz de costos locales, con el camino óptimo y las celdas sin soporte
  articular suficiente;
* el costo local en cada paso del camino.

Uso con datos locales (el HTML contiene coordenadas de pose y se guarda en la
carpeta de salida, fuera del repositorio)::

    python scripts/visor_alineamiento.py --caso <segment_id> --referencia <segment_id>

Demostración con secuencias sintéticas, sin datos de participantes::

    python scripts/visor_alineamiento.py --demo --archivo /tmp/visor_demo.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diana_comparativo.dtw import alineamiento_dtw, cargar_secuencia, distancia_dtw, validar_secuencia  # noqa: E402

ARISTAS_COCO17 = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                  (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4)]


def secuencias_demo() -> tuple[np.ndarray, np.ndarray]:
    """Dos ejecuciones sintéticas de distinta duración con articulaciones faltantes."""
    base = np.array([[0, -2.2], [-.1, -2.3], [.1, -2.3], [-.25, -2.2], [.25, -2.2], [-.5, -1.6], [.5, -1.6],
                     [-.7, -1.0], [.7, -1.0], [-.8, -.4], [.8, -.4], [-.3, 0], [.3, 0], [-.35, .8], [.35, .8],
                     [-.35, 1.6], [.35, 1.6]], dtype=float)

    def mover(n: int, fase: float) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n)[:, None]
        seq = np.repeat(base[None], n, axis=0)
        seq[:, [9, 10], 1] += 0.6 * np.sin(t + fase)
        seq[:, [7, 8], 1] += 0.3 * np.sin(t + fase)
        seq[:, [15], 0] += 0.4 * np.sin(t + fase)
        seq[:, [16], 0] -= 0.4 * np.sin(t + fase)
        return seq

    a, b = mover(30, 0.0), mover(44, 0.2)
    a[10:16, 9] = np.nan          # muñeca ausente en un tramo
    a[20:23, 5:11] = np.nan       # brazos ausentes: con la cabeza ausente en b, quedan < 9 articulaciones comunes
    b[5:28, :5] = np.nan          # cabeza ausente en un tramo largo
    return a.reshape(30, -1), b.reshape(44, -1)


def construir_html(caso: np.ndarray, referencia: np.ndarray, titulo: str, nota: str) -> str:
    resultado = alineamiento_dtw(caso, referencia)
    costos = np.where(np.isfinite(resultado["costos"]), resultado["costos"], -1.0)
    datos = {
        "titulo": titulo,
        "nota": nota,
        "distancia": resultado["distancia"] if np.isfinite(resultado["distancia"]) else None,
        "caso": np.where(np.isfinite(caso), caso, None).reshape(len(caso), -1, 2).tolist(),
        "referencia": np.where(np.isfinite(referencia), referencia, None).reshape(len(referencia), -1, 2).tolist(),
        "costos": np.round(costos, 4).tolist(),
        "camino": resultado["camino"],
        "aristas": ARISTAS_COCO17,
    }
    return PLANTILLA.replace("__DATOS__", json.dumps(datos, allow_nan=False))


PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visor de alineamiento DTW</title>
<style>
:root{--tinta:#1d1d1b;--suave:#5c5b56;--borde:#d9d8d3;--caso:#2a78d6;--ref:#eb6834;--fondo:#fcfcfb}
body{margin:0;font:14px/1.45 system-ui,sans-serif;color:var(--tinta);background:var(--fondo)}
main{max-width:1100px;margin:0 auto;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px}.nota{color:var(--suave);margin:0 0 16px}
.fila{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.panel{background:#fff;border:1px solid var(--borde);border-radius:8px;padding:12px}
.panel h2{font-size:14px;margin:0 0 8px}canvas{width:100%;height:auto;display:block}
.controles{display:flex;gap:12px;align-items:center;margin:12px 0}input[type=range]{flex:1}
button{font:inherit;padding:4px 12px;border:1px solid var(--borde);border-radius:6px;background:#fff;cursor:pointer}
.leyenda span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle}
.dato{font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1 id="titulo"></h1><p class="nota" id="nota"></p>
<p><b>Distancia DTW:</b> <span class="dato" id="distancia"></span>
<span class="leyenda"><span style="background:var(--caso)"></span>ejecución<span style="background:var(--ref)"></span>referencia</span></p>
<div class="controles"><button id="play">Reproducir</button><input id="paso" type="range" min="0" value="0">
<span class="dato" id="etiqueta"></span></div>
<div class="fila">
<div class="panel"><h2>Esqueletos alineados</h2><canvas id="esqueleto" width="520" height="420"></canvas></div>
<div class="panel"><h2>Matriz de costos y camino óptimo</h2><canvas id="matriz" width="520" height="420"></canvas>
<p class="nota">Gris: pares de fotogramas sin 9 articulaciones comunes (no admisibles).</p></div>
</div>
<div class="panel" style="margin-top:16px"><h2>Costo local a lo largo del camino</h2><canvas id="curva" width="1060" height="180"></canvas></div>
</main><script>
const D=__DATOS__;
const $=id=>document.getElementById(id);
$("titulo").textContent=D.titulo;$("nota").textContent=D.nota;
$("distancia").textContent=D.distancia===null?"sin alineamiento admisible (abstención)":D.distancia.toFixed(4).replace(".",",");
const camino=D.camino.length?D.camino:[[0,0]];const slider=$("paso");slider.max=camino.length-1;
function dibujarEsqueleto(ctx,puntos,color,dx){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=3;
 const p=puntos.map(q=>q[0]===null?null:[dx+q[0]*70,200+q[1]*70]);
 for(const [a,b] of D.aristas){if(p[a]&&p[b]){ctx.beginPath();ctx.moveTo(...p[a]);ctx.lineTo(...p[b]);ctx.stroke();}}
 for(const q of p){if(q){ctx.beginPath();ctx.arc(q[0],q[1],4,0,7);ctx.fill();}}}
const nI=D.costos.length,nJ=D.costos[0].length;let maximo=0;for(const f of D.costos)for(const v of f)maximo=Math.max(maximo,v);
const mc=$("matriz"),mx=mc.getContext("2d");const fondo=document.createElement("canvas");fondo.width=mc.width;fondo.height=mc.height;
(function(){const c=fondo.getContext("2d"),w=mc.width/nJ,h=mc.height/nI;
 for(let i=0;i<nI;i++)for(let j=0;j<nJ;j++){const v=D.costos[i][j];
  c.fillStyle=v<0?"#c9c8c3":`rgb(${Math.round(238-196*v/maximo)},${Math.round(244-150*v/maximo)},${Math.round(252-100*v/maximo)})`;
  c.fillRect(j*w,(nI-1-i)*h,w+1,h+1);}
 if(D.camino.length){c.strokeStyle="#1d1d1b";c.lineWidth=2;c.beginPath();
  D.camino.forEach(([i,j],k)=>{const x=(j+.5)*w,y=(nI-1-i+.5)*h;k?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();}})();
const cc=$("curva"),cx=cc.getContext("2d");
function dibujar(k){const [i,j]=camino[k];const e=$("esqueleto").getContext("2d");e.clearRect(0,0,520,420);
 dibujarEsqueleto(e,D.caso[i],"#2a78d6",150);dibujarEsqueleto(e,D.referencia[j],"#eb6834",370);
 mx.clearRect(0,0,mc.width,mc.height);mx.drawImage(fondo,0,0);const w=mc.width/nJ,h=mc.height/nI;
 mx.strokeStyle="#eb6834";mx.lineWidth=2;mx.strokeRect(j*w-2,(nI-1-i)*h-2,w+4,h+4);
 cx.clearRect(0,0,cc.width,cc.height);const vals=camino.map(([a,b])=>D.costos[a][b]);const top=Math.max(...vals,1e-9);
 cx.strokeStyle="#2a78d6";cx.lineWidth=2;cx.beginPath();vals.forEach((v,t)=>{const x=10+t*(cc.width-20)/Math.max(vals.length-1,1),y=170-150*v/top;t?cx.lineTo(x,y):cx.moveTo(x,y);});cx.stroke();
 const x=10+k*(cc.width-20)/Math.max(vals.length-1,1);cx.strokeStyle="#eb6834";cx.beginPath();cx.moveTo(x,10);cx.lineTo(x,175);cx.stroke();
 $("etiqueta").textContent=`paso ${k+1}/${camino.length} · fotograma ${i+1} ↔ ${j+1} · costo ${D.costos[i][j]<0?"n. a.":D.costos[i][j].toFixed(3).replace(".",",")}`;}
slider.oninput=()=>dibujar(+slider.value);let reloj=null;
$("play").onclick=()=>{if(reloj){clearInterval(reloj);reloj=null;$("play").textContent="Reproducir";return;}
 $("play").textContent="Pausa";reloj=setInterval(()=>{slider.value=(+slider.value+1)%camino.length;dibujar(+slider.value);},80);};
dibujar(0);
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rutas", default=None)
    parser.add_argument("--caso", help="identificador de la ejecución (segment_id)")
    parser.add_argument("--referencia", help="identificador de la referencia (segment_id)")
    parser.add_argument("--demo", action="store_true", help="usar secuencias sintéticas")
    parser.add_argument("--archivo", type=Path, help="ruta del HTML de salida")
    args = parser.parse_args()

    if args.demo:
        caso, referencia = secuencias_demo()
        html = construir_html(caso, referencia, "Demostración con secuencias sintéticas",
                              "Dos ejecuciones simuladas de distinta duración, con articulaciones faltantes.")
        destino = args.archivo or Path("visor_demo.html")
    else:
        if not (args.caso and args.referencia):
            parser.error("indique --caso y --referencia, o use --demo")
        from diana_comparativo.rutas import Rutas
        rutas = Rutas(args.rutas) if args.rutas else Rutas()
        rutas_npz = [rutas["secuencias"] / f"{sid}.npz" for sid in (args.caso, args.referencia)]
        for ruta in rutas_npz:
            validar_secuencia(ruta)
        caso, referencia = (cargar_secuencia(ruta) for ruta in rutas_npz)
        html = construir_html(caso, referencia, f"{args.caso} frente a {args.referencia}",
                              "Secuencias normalizadas (centrado en caderas, escala del torso).")
        destino = args.archivo or rutas.salida_de("visor") / f"{args.caso}__{args.referencia}.html"
        assert np.isclose(distancia_dtw(caso, referencia), alineamiento_dtw(caso, referencia)["distancia"], equal_nan=True)
    destino.write_text(html, encoding="utf-8")
    print(f"Visor escrito en {destino}")


if __name__ == "__main__":
    main()
