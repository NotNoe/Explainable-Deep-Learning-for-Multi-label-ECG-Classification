# Explainable Deep Learning for Multi-Label ECG Classification: Clinical Validation and Systemic Analysis of Signal Transformations

This repository contains the code used the paper named Explainable Deep Learning for Multi-Label ECG Classification: Clinical Validation and Systemic Analysis of Signal Transformations in the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB 2026).

## Authors
Noelia Barranco Godoy
Marian Diaz-Vicente
Sergio González-Cabeza
Mario Sanz-Guerrero
Belen Díaz Agudo
Juan A. Recio-Garcia

# Conference
ACM Conference on Bioinformatics, Computational Biology, and Health Informatics 2026

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


## Repository structure
- **final_models**: It contains pretrained models in hdf5 format. cwt_morlet model is divided in multiple parts (due to github size limitations). The original file can be recreated by executing the following commands:
    ```bash
    cat final_models/cwt_morlet_model/part_* > final_moddels/cwt_morlet_model.hdf5
    #Integrity check
    sha256sum -c final_models/cwt_morlet_moddel.sha256
    ```
- **ribeiro**: Contains [ribeiro's github repository](https://github.com/antonior92/automatic-ecg-diagnosis) code.
- **scripts**: Contains most of the scripts made for the work.
- **sergio**: Contains multiple scripts given by Sergio González Cabeza used in the preprocessing of the data.
- **test**: Contains the testing results for each model.
- **env.yaml**: The python environment used for the work.
- **perfects.json**: Output of `search_perfects.py`. It contains the line number and id of all testing ECKs considered "perfect", which means:
    - It has a real value of 1.0 for one of the classes and 0.0 for the others. This means that the ECK belongs to exactly one class with 100% confidence.
    - The model predicted correctly that the ECK belongs the correct class and only that one.