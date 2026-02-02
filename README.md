# DTDES-KGE

## Run this code on your data

Take FB15k-237 as example

### Quick Start

1、dowload all required data and checkpoint from：

**[Data and Checkpoint](https://drive.google.com/drive/folders/1dM1MU40oNKidDD8fD5f2RO9U6tkkQ2n-?usp=sharing)**

Attention: If the downloaded file is named checkpoint.zip, please do not extract it. Instead, rename it to checkpoint by deleting the .zip extension.


2、Run the code

```
python main.py config/FB15k-237_0.json
```



### Train from scratch

1、Trian Teacher Model



(1) Take LorentzKG-FB15k-237 as example

```
cd TeacherModels
```

```
cd LorentzKG
```

```
bash run_fb15k.sh
```



(2)  Offical implement of Teacher Model：

**[LorentzKG](https://github.com/LorentzKG/LorentzKG)**

**[RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)**

**[HAKE](https://github.com/MIRALab-USTC/KGE-HAKE)**



2、Pre-distill phase

```
"execute_ditill_teacher_to_rotate": true
```



3、Get query-aware triples for each query

```
"only_get_qtdict": true
```



4、Curvature_Estimation

please refer to：https://github.com/colab-nyuad/Curvature_Estimation



## Acknowledgments

We sincerely thanks to the following open-source repository:

**[DualDE](https://github.com/YushanZhu/DistilE)**

**[IterDE](https://github.com/seukgcode/IterDE)**

**[LorentzKG](https://github.com/LorentzKG/LorentzKG)**

**[MRME-KGC](https://github.com/2391134843/MRME-KGC)**

**[RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)**

**[HAKE](https://github.com/MIRALab-USTC/KGE-HAKE)**

**[GIE](https://github.com/Lion-ZS/GIE)**

**[Curvature_Estimation](https://github.com/colab-nyuad/Curvature_Estimation)**

**[AttH](https://github.com/tensorflow/neural-structured-learning)**