# Bitácora de Desarrollo
**Proyecto:** 3D-HepaticNuclei-Classifier   
**Tesis:** Clasificador automático de Núcleos Celulares en Imágenes 3D de Tejido Hepático utilizando 3DINO  
**Año:** 2026  

---

## 1) Validación inicial de Bounding Boxes

Durante la etapa inicial de validación de los parches 3D se detectó un problema en la generación de bounding boxes: los “cuadros” se estaban generando con tamaño **12** y no con **13**, lo que provocaba inconsistencias dimensionales en el pipeline.

**Acción correctiva:**
- Se ajustó la configuración para que los bounding boxes se generen con tamaño **13**.
- Se verificó nuevamente la consistencia dimensional de los parches.
- Se planificó documentar esta validación en un notebook específico de verificación.

---

## 2) Integración del modelo 3DINO desde GitHub

Un desafío importante fue integrar correctamente el modelo **3DINO** desde el repositorio en GitHub, ya que inicialmente no se comprendía la estructura del proyecto ni la forma correcta de cargar la configuración y los pesos preentrenados.

**Dificultades encontradas:**
- Comprensión de la organización del repositorio y sus dependencias.
- Manejo de rutas y configuración dentro de Google Colab/Drive.
- Carga correcta del modelo en modo evaluación.

**Solución implementada:**
- Se configuraron rutas explícitas para el archivo de configuración y el checkpoint de pesos.
- Se cargó el modelo en `eval()` y se verificó el dispositivo (GPU/CPU).

---

## 3) Preprocesamiento de datos

La etapa de preprocesamiento presentó dificultades, principalmente porque existían múltiples alternativas posibles y no era claro cuál estrategia era la más adecuada para este dataset.

**Dificultades encontradas:**
- Selección de transformaciones de preprocesamiento.
- Problemas con el **padding** (no siempre se aplicaba correctamente).
- Normalización en rango **[-1, 1]**, que en algunos casos podría estar alterando la distribución o introduciendo efectos no deseados.

**Acciones actuales / próximas:**
- Probar normalización alternativa en rango **[0, 1]** para comparar estabilidad.
- Revisar y validar que el padding sea consistente en todos los casos.

---

## 4) Balanceo por Data Augmentation

El análisis actual se realizó con un esquema de balanceo agresivo mediante data augmentation, generando **7000 muestras aumentadas por clase**. Sin embargo, los resultados no mostraron mejoras claras frente al baseline.

**Riesgos/limitaciones detectadas:**
- No se realizó verificación visual sistemática de las transformaciones, por lo que podrían existir aumentos poco realistas.
- El sobremuestreo hasta miles de ejemplos por clase puede generar redundancia (muestras muy similares) y afectar la generalización.

**Experimentos planificados:**
- Ejecutar el pipeline **sin augmentation**.
- Comparar distintos tamaños por clase: **350**, **500**, **3500**, **7000** y **7500**.
- Evaluar aumentos específicos: **Gaussian Noise** y **Gaussian Blur**.

---

## 5) Dificultades de discriminación entre clases

Los resultados muestran patrones consistentes con dificultades de discriminación, especialmente entre las clases **1–2–3–4**, y un desempeño particularmente bajo en la clase **3**.

**Interpretación preliminar:**
- Puede existir solapamiento morfológico real entre estas clases.
- Es posible que el espacio de embeddings no sea suficientemente separable para esas categorías.
- La optimización de hiperparámetros no mejoró significativamente, lo que sugiere un límite más relacionado a representación/datos que al clasificador.

---

## 6) Ablación propuesta: excluir la clase “Other”

Si después de los nuevos experimentos (preprocesamiento, normalización y augmentation) el rendimiento continúa siendo bajo, se propone un experimento adicional:

**Acción propuesta:**
- Excluir la última clase (“Other”) y repetir el análisis con únicamente 4 clases principales.

**Objetivo:**
- Evaluar si la heterogeneidad de la clase “Other” está degradando la separabilidad global y afectando el rendimiento del clasificador.

---

## 7) Próximos pasos

- Probar normalización **[0, 1]** vs **[-1, 1]**.
- Verificar visualmente las transformaciones de augmentation de forma sistemática.
- Ejecutar experimentos comparativos por tamaño de dataset balanceado (350 / 500 / 3500 / 7000 / 7500).
- Incorporar **Gaussian Blur** y **Gaussian Noise** como aumentos controlados.
- Si no hay mejora, repetir experimentos excluyendo la clase “Other”.
- Complementar el análisis con visualización exploratoria de embeddings (PCA / t-SNE).

---
