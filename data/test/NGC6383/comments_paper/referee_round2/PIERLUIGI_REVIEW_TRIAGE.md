# Revisión de Pierluigi (round 2) — inventario y triaje

_Preparado 2026-08-16 como triaje previo. **TODO APLICADO Y VERIFICADO — ver §5 al final.**
Las secciones 1–4 se conservan tal como se escribieron, incluidas las decisiones que después
cambiaron (T2/T3 pasaron de "rechazar" a "mover a `\tablefoot`" al leer la guía de A&A), porque el
razonamiento descartado es parte del registro. §5 dice qué quedó realmente en el `.tex`._

⚠ Este encabezado decía "Nada aplicado todavía" hasta después de aplicarlo todo. Se corrige aquí
porque es exactamente el modo de fallo que este documento registra: **una aserción que fue cierta
cuando se escribió y dejó de serlo sin que nadie la revisara.**

Fuentes:
- PDF anotado: `~/Downloads/aanda_revised_clean-1.pdf` (23 pp., **41 anotaciones**)
- Cover letter devuelta: `~/Downloads/cover_letter_round2.txt`
- Manuscrito: `submission_package/clean_source/aanda.tex` (607 líneas, commit `80adcc6`, 2026-07-19)
- Respuesta: `submission_package/letters/response_to_referee_round2.txt` (R1–R17)

---

## 1. Cover letter — 2 cambios, ambos triviales

Diff contra el repo. **El repo tiene el original; el archivo de Downloads trae las correcciones**
(confirmado por `git log -S`: la forma del repo es la del 19-jul y "their" nunca existió en el
historial).

| # | original (repo) | corrección de Pierluigi |
|---|---|---|
| 1 | "We agree with **its** central criticism" | "We agree with **their** central criticism" |
| 2 | "…accordingly**:** the paper is now structured" | "…accordingly**.** The paper is now structured" |

Ambos son de estilo. #1 cambia el referente de *the report* a *the referee* — y "their" singular
es además la forma neutra, coherente con cómo firmas. Sin objeción.

**Acción:** copiar el archivo de Downloads sobre `submission_package/letters/cover_letter_round2.txt`.

---

## 2. Las 41 anotaciones del PDF

| tipo | n | naturaleza |
|---|---|---|
| Highlight **con nota** | 11 | correcciones de redacción |
| StrikeOut (sin nota) | 30 | texto que considera redundante |

### 2.0 Verificación definitiva (segunda pasada, 2026-08-16)

⚠ **La primera pasada localizó las anotaciones con un matcher de palabras, y ese método no puede
fallar en anotaciones cortas**: con una sola palabra marcada (`structure.`), *cualquier* línea que
la contenga puntúa 1.00. Reportó "39/41 mapeadas con ≥60%", cifra que para las cortas no
significaba nada. Rehecho extrayendo el contexto real de la página alrededor de cada rectángulo.
**Dos mapeos estaban mal** y aparecieron **dos multiplicidades** que la primera pasada no vio.

Recuento exacto de los 30 tachados: **28 en captions, 2 en cuerpo** (l.391 y l.400). De los 28 en
captions, 3 son los caveats científicos de la Tabla 1 (l.77). Total **25 verbosidad pura + 5 de
contenido** — la cifra original se sostiene.

### 2.1 Los 11 comentarios de forma — todos aceptables

Con la línea **verificada** del `.tex` (no la del matcher):

| # | p. | línea | pide | veredicto |
|---|---|---|---|---|
| 1 | 4 | **151** cuerpo | `outer structure` → `structures` | ✅ trivial |
| 2 | 5 | **193** cuerpo | `posterior` → `posterior estimate` | ✅ trivial |
| 3 | 5 | **209** cuerpo | punto suelto: `. and the optimal KDE bandwidth.` | ✅ **es una errata real** |
| 4 | 8 | **265** cuerpo | `correspond approximately` → `approximately correspond` | ✅ trivial |
| 5 | 11 | **351** cuerpo | `depends directly` → `directly depends` | ✅ trivial |
| 6 | 12 | **359** cuerpo | `early-dynamical` → `early dynamical` | ⚠ **ver multiplicidad** |
| 7 | 13 | **387** cuerpo | `;` → `.` antes de `Kalari (2019) notes` | ✅ y coincide con §3/§3bis |
| 8 | 13 | **389** cuerpo | comprimir a `(0.94 probability of being a binary)` | ✅ |
| 9 | 14 | **400** cuerpo | `compatible` → `consistent` | ✅ mejor uso estadístico |
| 10 | 14 | **406** cuerpo | `; in line` → `. In line with` | ✅ |
| 11 | 21 | **590** caption Ap. D | `fractional parallax error below 0.1` → `δϖ/ϖ < 0.1` | ✅ **y refuerza: ver abajo** |

**Correcciones a la primera pasada:**
- **#11 NO está en el caption de la Tabla 1 (l.77)** como dije, sino en el **caption de la tabla del
  Apéndice D (l.590)**.
- **#6 no está en el abstract (l.35)** como dije, sino en el cuerpo (l.359).

**Dos multiplicidades que la primera pasada no vio — marcó una ocurrencia, hay cinco:**

| término | ocurrencias | implicación |
|---|---|---|
| `early-dynamical` | **l.35, 353, 359, 361, 412** | quitar el guion en una sola deja el manuscrito inconsistente. Aplicar a las 5 o a ninguna. |
| `fractional parallax error` (en prosa) | **l.77, 177, 179, 186, 590** | el `.tex` **ya usa `$\delta\varpi/\varpi$` en l.186**, así que lo de Pierluigi no es preferencia sino **consistencia**: hay dos notaciones conviviendo. Refuerza aceptarlo, y conviene decidir si l.77 también cambia. |

Ninguno toca ciencia. El de p.13 es el **mismo pasaje de Kalari** que señala en el correo (§3).

### 2.2 Los 30 tachados — 25 seguros, 5 en conflicto

**25 son verbosidad de pies de figura** (colores, símbolos, mapas *viridis*, barras de escala,
repeticiones del cuerpo). Ejemplos: `(viridis color map; color bar at right)`, `A scale bar of 5
parsecs is included at the bottom right…`, la enumeración de cuartiles en Figs. 10–11, la lista de
radios con sus colores en Fig. 20. **Todos aceptables** — es exactamente la redundancia que dijo
estar cortando.

**5 tocan contenido científico.** Aquí aplica su propia instrucción: *"ignora las eliminaciones de
texto que afecten las modificaciones hechas para abordar los comentarios del/la evaluador/a."*

| # | p. | texto tachado | veredicto |
|---|---|---|---|
| **T1** | 3 | `…95% credible-interval bounds of the Beta posterior, **not the 1σ default**` (caption Tabla 1) | ⚠ **Discutible.** "95% credible-interval bounds" ya lo dice; el contraste es redundante *estrictamente*. Pero el caption entero existe porque **R5** pidió aclarar la Tabla 1. Cortar aquí es de bajo riesgo. |
| **T2** | 3 | `…y no es independiente del valor basado en paralaje porque el prior del módulo de distancia estaba centrado en él` | 🚫 **RECHAZAR.** Esto **es la respuesta a R5** — el referee preguntó literalmente qué es "Distance (D.M.)" y si deriva de Bailer-Jones. La cláusula de no-independencia es la advertencia científica que responde eso. |
| **T3** | 3 | `…la única extracción cuyo perfil alcanza un anillo de fondo ajustado más allá del cúmulo; todas las ventanas más angostas dan R_t consistentes` | 🚫 **RECHAZAR.** Esto **es la respuesta a R11**, el único cambio científico mayor de la ronda. Todo el argumento del radio de extracción descansa en esa frase; está en el caption, en §5 (l.242), en §8.4 (l.400) y en el Apéndice D (l.563). Borrarla del caption deja la Tabla 1 sin justificar por qué R_c y R_t vienen de ajustes distintos. |
| **T4** | 13 | `…usamos sus offsets de color R_C−Hα publicados, 0.12–0.24 mag sobre la relación de secuencia principal para candidatos y >0.24 mag para emisores` | ⚠ **Revisar.** Es transparencia metodológica: explica que **no** recuperamos la lista de emisores de Rauw sino que usamos sus offsets. Ligado a **R16** (organización y procedencia). Si se corta, hay que reubicarlo, no perderlo. |
| **T5** | 14 | lista de radios de la literatura: `15.0, Kharchenko 2005; 29.0±4.2, Piskunov 2008; 22.7, Hunt & Reffert 2024; 29.7±7.7, Pulgar-Escobar & Henríquez-Salgado 2024` | ⚠ **Discutible.** Responde a **R11** ("comparación con los radios menores de la literatura"). Los valores están también en la Tabla `tab:literature`, así que el paréntesis *es* redundante — pero conviene dejar el rango "15–30 arcmin" y la referencia a la tabla. |

> **Nota transversal:** los tres tachados de la p.3 caen dentro del **mismo caption de la Tabla 1**
> (l.77), que ocupa ~15 líneas. La queja de fondo (es demasiado largo) es legítima. La salida
> correcta no es borrar los caveats sino **moverlos al cuerpo** — R5 y R11 quedan respondidos igual
> y el caption se aliviana.

---

## 3. La frase de Kalari — Pierluigi tiene razón, y hay más

Su objeción, sobre la **carta de respuesta** (R16):

> "Kalari (2019) interpolated H-alpha-selected CTTS on PARSEC isochrones and reports that the
> identical data yield mean ages of 2.9 and 3.5 Myr with a median age of 2.8 Myr **the** Tognelli
> et al. (2011) and Siess et al. (2000) models"

**Tres defectos, no uno:**
1. Falta la preposición antes de `the Tognelli` — la cláusula `with a median age of 2.8 Myr` se
   insertó dentro de `with … the … models` y la dejó huérfana.
2. Se lee como si 2.9, 3.5 **y** 2.8 salieran de Tognelli+Siess: tres números, dos modelos.
3. Mezcla *mean* con *median* sin señalarlo.

**El manuscrito (l.387) está bien redactado** y dice lo correcto:
> "…PARSEC models reddened by E(B−V)=0.32; Kalari (2019) notes that the same data yield mean ages
> of 2.9 and 3.5 Myr with the Tognelli (2011) and Siess (2000) models"

Es decir: **la carta es una paráfrasis rota de una frase correcta.** El arreglo se limita a la carta.

### 3.1 Pero verificando la fuente aparece un problema anterior

Leído Kalari 2019 (arXiv:1901.07511) directamente:

| lugar | qué dice |
|---|---|
| **Abstract** | "has a **median age** of 2.8 ± 1.6 Myr" |
| **§3.3 (cuerpo)** | "The **mean ages** … from the Bressan (2012) isochrones is **2.8 ± 1.6 Myr**" |
| **§3.3** | "The **mean** age … Tognelli (**2.9 ± 1.7**) and Siess (**3.5 ± 2.4**) … is higher" |

**Kalari se contradice a sí mismo**: el mismo 2.8 ± 1.6 es *median* en el abstract y *mean* en el
cuerpo. Nuestro manuscrito heredó la contradicción — cita 2.8 como **median** (siguiendo el
abstract) y 2.9/3.5 como **mean** (siguiendo el cuerpo), de modo que **compara una mediana contra
dos medias**. Eso es justo la incoherencia que Pierluigi olfateó en la carta, pero el origen está
en el manuscrito y en el paper de Kalari.

**Lo defendible:** los tres son **medias** según §3.3 (2.8 ± 1.6 Bressan/PARSEC, 2.9 ± 1.7
Tognelli, 3.5 ± 2.4 Siess). El "median" del abstract es un desliz del propio Kalari.

**Además** el manuscrito omite las incertidumbres de 2.9 y 3.5 (±1.7 y ±2.4), que sí están en la
fuente y que **importan**: con esas barras, la diferencia entre familias de modelos deja de ser un
efecto claro — que es exactamente el punto que la frase quiere hacer.

⚠ **Esto toca la Tabla `tab:literature`** (l.467), donde figura `2.8±1.6` bajo Kalari. Hay que
decidir si se rotula *mean* y si se decide una sola vez para manuscrito, tabla y carta.

Verificado también contra la fuente y **correcto** en el manuscrito: 55 CTTS cinemáticos, masas
0.3–1.0 M☉, 94% en el locus T-Tauri NIR, selección por EW(Hα) + doble gaussiana en movimiento
propio. (Dato adicional no usado: Kalari adopta **d = 1340 pc**, frente a nuestros 1.11 kpc.)

---

---

## 3bis. SEGUNDA PASADA sobre Kalari — cuatro cosas más, una de ellas grave

Releído el cuerpo completo de Kalari 2019 §3.3. La primera pasada se quedó corta.

### (a) El ±1.6 es el **error de la media**, no la dispersión

> "The error on the mean age takes into account the reddening uncertainties and photometric errors.
> **The standard deviation of CTTS ages is 3.4 Myr**, and 70% of all CTTS have ages between 1–4 Myr."

La distribución real: rango **0.4–18 Myr**, IQR **1.4–3.8**, **σ = 3.4 Myr**. Nuestro `2.80 ± 1.60`
está mal por partida doble: no es mediana (es media) y el ±1.6 no es una dispersión sino el error
de esa media. Citarlo como si fuera la anchura de la población hace parecer a la población de
Kalari **el doble de estrecha de lo que es**.

### (b) El rango de masas también difiere entre abstract y cuerpo

| | masas |
|---|---|
| Abstract de Kalari | "mass range between **0.3 and 1** M☉" |
| §3.3 de Kalari | "The mass range is between **0.3–0.9** M☉, **with the median mass 0.5 M☉**" |
| **Nuestro l.387** | "masses ranging from 0.3 to **1.0** M☉" ← seguimos el abstract otra vez |

Mismo patrón que con mean/median: **dos veces tomamos el abstract donde el cuerpo dice otra cosa.**

### (c) 🔴 Kalari adopta **d = 1340 pc**; nosotros 1.11 kpc — y no lo decimos en ninguna parte

`grep` sobre `aanda.tex`: **cero menciones** de la distancia de Kalari.

| | |
|---|---|
| DM Kalari (1340 pc) | 10.636 |
| DM nuestro (1110 pc) | 10.227 |
| **Δ** | **0.41 mag → factor 1.46 en luminosidad** |

Kalari *interpola posiciones del CMD sobre tracks PMS*, así que la distancia entra directamente en
la edad. A mayor distancia supuesta, las mismas estrellas son intrínsecamente **más luminosas**, y
sobre una track de contracción PMS eso significa **más jóvenes**. Es decir: **sus edades están
sesgadas hacia lo joven respecto de las nuestras por construcción**, en la dirección que explicaría
parte de la diferencia con nuestros 3.53 Myr.

Y nuestro texto actual dice que la diferencia *"reflects the choice of evolutionary code **rather
than the data**"*. Con una diferencia de distancia del 21% eso es, en el mejor de los casos,
incompleto: la distancia **es** un dato de entrada, y difiere. Un referee que conozca a Kalari
puede objetar esa frase.

⚠ No afirmar cuántos Myr son sin calcularlo sobre las tracks: el signo es firme, la magnitud no.

### (d) Kalari ya trae el mecanismo y el orden CTTS/WTTS — y responde a R16

- **Por qué difieren los modelos**, con cita: sigue el patrón de **Herczeg & Hillenbrand (2015)** —
  para tipos anteriores a M (el grueso de su muestra) Bressan da edades menores que Siess. Nuestro
  texto dice vagamente "the choice of evolutionary code"; hay una referencia y una dirección.
- **Comparación con Rauw hecha por él mismo**: Rauw da 2.8 ± 0.5 Myr para las estrellas X, y Kalari
  advierte que *"X-ray emitting PMS stars (likely Weak Line T Tauri stars) are generally older than
  CTTS"*. **R16 era precisamente sobre cómo ordenamos la comparación Kalari↔Rauw** — y el propio
  Kalari ya explica por qué esos dos números no son directamente comparables.
- Kalari advierte además (citando Mayne & Naylor 2008) que la edad **estadística de la población**
  es más fiable que las edades individuales — relevante porque nosotros comparamos una edad de
  cúmulo contra un estadístico de población con σ = 3.4 Myr.

---

## 3ter. Opciones para los tres tachados discutibles

**El hallazgo que las ordena:** la guía de A&A dice que las tablas deben ser autoexplicativas,
pero que *"details should not clutter the header and are better presented as **explanatory
footnotes**"*. El caption de la Tabla 1 tiene **233 palabras**, y `\tablefoot{}` **ya se usa en
este mismo manuscrito** (l.170). Eso da una salida que no había considerado y que no es un
compromiso: no es "borrar" ni "dejar", es **mover a footnote**.

### T1 — "not the 1σ default" (Y_frac)

| opción | a favor | en contra |
|---|---|---|
| **A. Aceptar el corte** | "95% credible-interval bounds" ya lo dice | pierde el contraste explícito con el resto de la tabla |
| **B. Mover a `\tablefoot`** ⭐ | conserva la precisión, alivia el caption, forma nativa A&A | ninguno |

### T4 — offsets de color de Rauw (0.12–0.24 / >0.24 mag)

| opción | a favor | en contra |
|---|---|---|
| **A. Mantener** | es procedencia metodológica, ligada a R16 | es la frase más densa del párrafo |
| **B. Comprimir en sitio** ⭐ | conserva los números y la lógica "no recuperamos su lista de emisores" | requiere reescribir |
| **C. A footnote** | máximo alivio | rompe el hilo del argumento, que es prosa y no dato tabular |

Recomiendo **B**: el referee (R16) pidió *procedencia*; borrar cómo se hizo el cruce va en contra.

### T5 — lista de radios de la literatura (15.0; 29.0±4.2; 22.7; 29.7±7.7)

| opción | a favor | en contra |
|---|---|---|
| **A. Aceptar el corte** ⭐ | los cuatro valores **están en `tab:literature`**; el paréntesis es redundancia real, y el rango "15–30 arcmin" + la referencia a la tabla sobreviven | el lector pierde los valores en línea |
| B. Mantener | autocontenido | duplica la tabla |

Éste es el único de los tres donde Pierluigi tiene razón sin matices.

### Y para T2/T3 (los que había marcado como rechazar)

Con el mecanismo de footnote, **deja de ser rechazo**: el caveat de no-independencia (R5) y el del
anillo de fondo (R11) **se mueven a `\tablefoot`**. Pierluigi obtiene el caption corto que pide,
el referee conserva sus respuestas, y A&A obtiene su forma preferida. Es la opción que satisface a
los tres.

---

## 4. Qué falta decidir antes de tocar nada

1. **T2 y T3 se rechazan** — deshacen R5 y R11. Confirmar con Pierluigi que su instrucción cubre
   estos dos casos (creo que sí: son literalmente cambios hechos para el referee).
2. **El caption de la Tabla 1** necesita adelgazarse **moviendo** contenido al cuerpo, no borrándolo.
   Esa es la manera de darle a Pierluigi lo que pide sin perder las respuestas al referee.
3. **Mean vs median en Kalari**: decidir una sola convención y aplicarla a l.387, a la Tabla
   `tab:literature` (l.467) y a R16. Sugerido: llamarlos a los tres *mean* y añadir ±1.7 y ±2.4.
4. **T4 y T5**: decidir si se recortan o se reubican.
5. El límite de 12 páginas sigue abierto — estos cortes ayudan, y conviene contar cuánto ahorran
   antes de discutir con A&A.

---

## 5. VERIFICACIÓN FINAL — 2026-08-16, releído contra el `.tex` aplicado

Cada punto comprobado **en contexto** sobre `clean_source/aanda.tex`, no por presencia de palabra.

📁 **Las cartas que se envían viven en `submission_package/letters/`.** Las copias de este
directorio (`cover_letter_round2.txt`, `response_letter.txt`) son de trabajo y llegaron a divergir
de las buenas; se resincronizaron el 2026-08-16. Ante duda, manda `submission_package/letters/`.

### Los 11 comentarios: 10 aplicados + 1 ya satisfecho

⚠ **Corregido 2026-08-16.** Esta línea decía **11/11 aplicados** y contaba C2 como aplicado. No lo
está, y no debe estarlo. La anotación literal es
`{"page": 5, "type": "Highlight", "comment": "posterior estimate", "text": "posterior"}`: pide
`posterior` → `posterior estimate`. La cadena `posterior estimate` **nunca ha existido** en el
`.tex` (cero ocurrencias antes y después de toda edición). Lo que ocurrió es que el resumen
reformuló la petición como "`posterior distribution` (no *estimate*)" y, así reformulada, la dio
por cumplida — porque el texto ya decía `posterior distribution`.

Veredicto tras releer el sitio: el `posterior` de la p.5 es el del ajuste de King
("yielding the posterior distribution of the structural parameters"), y **ese sampler sí
converge**, así que la palabra es correcta y `posterior estimate` sería menos preciso, no más.
**No se cambia el manuscrito; se corrige el recuento.** La lección es la del propio documento: un
resumen que reformula la petición puede declararla cumplida sin que nadie haya tocado nada.

C1 `outer structures` · C2 **no aplicado, ya satisfecho** (ver arriba) · C3 puntuación KDE ·
C4 `approximately correspond` · C5 `directly depends` · C6 `early dynamical` **×6** · C7 punto antes
de `Kalari (2019) notes` · C8 `(0.94 probability…)` comprimido · C9 `consistent` · C10 punto antes
de `In line with` · C11 `$\delta\varpi/\varpi<0.1$`.

⚠ C10 apareció como fallo en el primer chequeo automático — **falso negativo**: mi patrón usaba
`~\ref` y el texto tiene ` \ref`. La edición estaba bien; el verificador estaba mal. Es el mismo
modo de fallo que este documento viene registrando: **un chequeo que falla por su propia forma, no
por el hecho**.

### Los 30 tachados: 30/30 resueltos

| grupo | n | resultado |
|---|---|---|
| verbosidad de captions | 24 | cortados (verificado: sin `viridis`, sin barras de escala, sin "As shown in the figure", sin la enumeración de columnas del CDS) |
| caption tabla literatura | 1 | `gathered from various studies` cortado |
| **T1** `not the 1σ default` | 1 | cortado |
| **T2** no-independencia (**R5**) | 1 | **preservado** en `\tablefoot` |
| **T3** anillo de fondo (**R11**) | 1 | **preservado** en `\tablefoot` |
| **T4** offsets de Rauw | 1 | comprimido, números intactos |
| **T5** lista de radios | 1 | cortado (los 4 valores siguen en `tab:literature`) |

Caption de la Tabla 1: **233 → 32 palabras**, con `\tablefoot` presente.

Comprobado además que las anulaciones del agente **no perdieron identificaciones**: el mapeo de
símbolos del CMD, el umbral PMS, la distinción de ventana de 40′ (R11), la isócrona de
`fig:mass_binary` y el nivel de fondo `b` siguen todos en sus captions.

### Kalari: 7/7

mean en vez de median · el ±1.6 declarado como error de la media con σ = 3.4 · masas 0.3–0.9 con
mediana 0.5 · incertidumbres ±1.7 y ±2.4 · la distancia de 1.34 kpc y el sesgo que implica ·
Herczeg & Hillenbrand citado (bibcode verificado en ADS) · celda de distancia rellenada en la tabla.
La cláusula *"rather than the data"* ya no está.

### Cartas: 5/5

Las dos correcciones de Pierluigi en la cover letter; la frase rota eliminada de la respuesta; y la
respuesta ahora **declara al referee** las cuatro correcciones de Kalari y el sistemático de
distancia, en vez de callarlas.
