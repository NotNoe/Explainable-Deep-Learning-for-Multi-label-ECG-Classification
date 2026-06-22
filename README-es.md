# Predicción de Riesgo Cardiovascular Explicable Mediante Deep Learning

Este repositorio contiene el código utilizado para el desarrollo de mi Trabajo de Fin de Grado del **Doble Grado en Matemáticas e Informática** de la **Facultad de Informática** de la **Universidad Complutense de Madrid**.

## Título
Predicción de riesgo cardiovascular explicable meddiante deep learning.

Explainable Cardiovascular Risk Prediction using Deep Learning

## Autor
Noelia Barranco Godoy
## Directores
Belén Díaz Agudo
Juan A. Recio García
María Ángeles Díaz Vicente

## Resumen
Las enfermedades cardiovasculares son una de las principales causas de muerte
según la OMS, para combatir esto, una de las mejores herramientas diagnósticas son
los electrocardiogramas (ECGs). Recientemente se han desarrollado algunos modelos
Deep Learning que permiten clasifcar de manera bastantee efciente los ECGs, pero
los médicos se han mostrado reacios a utilizarlos debido a que estos modelos no
suelen ser explicables.

En este trabajo se modifca uno de los modelos más conocidos en el la literatura
(Ribeiro et al., 2020) para la predicción y clasifcación de anomalías cardíacas a partir
de señales de electrocardiogramas (ECG), integrando técnicas de explicabilidad que
permiten justifcar las decisiones del modelo. Utilizando la base de datos pública
PTB-XL, se implementaron transformaciones de señal, como la Transformada de
Fourier de Tiempo Reducido (STFT) y la Transformada de Onda Continua (CWT),
con el objetivo de extraer características de tiempo y frecuencia para mejorar las
predicciones.

Los resultados obtenidos indican que el modelo base, logra un desempeño desta-
cado en la clasifcación multiequeta de anomalías cardíacas. Sin embargo, las trans-
formaciones de señal presentan limitaciones cuando se utilizan con esta arquitectu-
ra. Para abordar la necesidad de transparencia en el ámbito médico, se emplearon
saliency maps, validados por expertos médicos, quienes destacaron su potencial edu-
cativo y clínico, señalando áreas de mejora para futuras aplicaciones.

El estudio concluye que la combinación de técnicas de deep learning y métodos
de explicabilidad puede contribuir signifcativamente a la adopción de herramientas
de IA en entornos médicos, optimizando el diagnóstico temprano y apoyando la
formación de profesionales de la salud.

## Abstract
Cardiovascular diseases are one of the leading causes of death according to the
WHO. The electrocardiogram (ECG) is one of the most efective diagnostic tools
to address this issue. Recently, some eep learning models have been developed
to efciently classify ECG's, but medical professionals have been reluctant to adopt
them due to the lack of explainability in these models.

In this study, one of the most well-known models in the literature (Ribeiro et
al., 2020) for predicting and classifying cardiac anomalies from electrocardiogram
(ECG) signals is modifed by integrating explainability techniques to justify the
model's decisions. Using the publicly available PTB-XL database, signal transfor-
mations such as the Short-Time Fourier Transform (STFT) and Continuous Wavelet
Transform (CWT) were implemented to extract time and frequency features with
the goal of improving predictions.

The results indicate that the base model achieves outstanding performance in
the multi-label classifcation of cardiac anomalies. However, signal transformations
demonstrate limitations when used with this architecture. To address the need for transparency in the medical field, saliency maps were employed and validated
by medical experts, who highlighted their educational and clinical potential while
identifying areas for improvement in future applications.

The study concludes that the combination of deep learning techniques and explainability methods can signifcantly contribute to the adoption of AI tools in med-
ical settings, enhancing early diagnosis and supporting the education of healthcare
professionals.


## Estructura del repositorio
- **final_models**: Contiene los modelos ya entrenados en formato hdf5. El modelo "cwt_morlet", por limitaciones de tamaño en github, está dividido en varias partes. Para recomponerlo simplemente ejecutar el siguiente comando:
    ```bash
    cat final_models/cwt_morlet_model/part_* > final_moddels/cwt_morlet_model.hdf5
    #Para comprobar la integridad del archivo
    sha256sum -c final_models/cwt_morlet_moddel.sha256
    ```
- **ribeiro**: Contiene código del [repositorio de github de ribeiro](https://github.com/antonior92/automatic-ecg-diagnosis).
- **scripts**: Contiene la mayoría de scripts propios que he desarrollado para varias partes del trabajo.
- **sergio**: Contiene varios scripts facilitados por Sergio González Cabeza para el preprocesamiento del trabajo.
- **test**: Contiene los resultados de testing de cada modelo.
- **env.yaml**: El volcado del entorno de conda con el que se ha ejecutado el software.
- **perfects.json**: La salida del script `search_perfects.py`. Contiene la línea y el id de todos los ECGs del subset de testing que son predicciones perfectas para cada clase. Una predicción se considera perfecta si cumple lo siguiente:
    - Tiene valor real de 1.0 para una clase y 0.0 para las demás, es decir, el ECG pertenece exclusivamente a una clase, con un 100% de seguridad.
    - La única etiqueta predicha por el modelo es el de la clase correcta. Es decir, que el modelo ha predicho correctamente que el ECG pertenece exclusivamente a esa clase.
- **TFG.pdf**: El pdf con el trabajo.
