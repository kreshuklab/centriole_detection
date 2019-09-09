# Synthetic patches, real images: screening for centrosome aberrations in EM images of human cancer cells

Recent advances in high-throughput electron microscopy imaging enable detailed study of centrosome aberrations in cancer cells.
While the image acquisition in such pipelines is automated, manual detectionof centrioles is still necessary to 
select cells for re-imaging at higher magnification.




In this repository we propose an algorithm which performs
this step automatically and with high accuracy. From the image labels produced by human experts and a 3D model of 
a centriole we constructan additional training set with patch-level labels. 
A two-level DenseNetis trained on the hybrid training data with synthetic patches and realimages, 
achieving much better results on real patient data than trainingonly at the image-level.

## Reposirory structure
There are 8 scripts in the root directory covering all our experiments.
Each script has name "run_*.py" and starts the training process of a model.

Implemented architectures for our experiments are stored in file [src/implemented_models.py](https://github.com/kreshuklab/centriole_detection/blob/master/src/implemented_models.py).
To reproduce results described in our paper one should train 2 models: for patches and full-images.
The most convenient way is to use the script [run_ilc_1ch.py](https://github.com/kreshuklab/centriole_detection/blob/master/run_ilc_1ch.py).

For training patch lavel model:
```bash
run_ilc_1ch.py --model_name ICL_DenseNet_3fc
```

For training patch lavel model:
```bash
run_ilc_1ch.py --model_name MIL_DenseNet_3fc_bn --rdt --img_size 512 --init_model_path <path_to_patch_level_model>
```
Due to personal data security we can not make our dataset public.
**Do not forget to change paths in the script for your data**.
Currently, repository is not well organized, so in case something is unclear do not hesitate to contact [Artem Lukoyanov](https://github.com/ottogin) and any help is welcome.

## Publication

This work has been accepted to MICCAI 2019 and you can find a [preprint on arXiv](https://arxiv.org/abs/1908.10109).
