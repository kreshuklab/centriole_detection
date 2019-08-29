# Synthetic patches, real images: screening for centrosome aberrations in EM images of human cancer cells

Recent advances in high-throughput electron microscopy imaging enable detailed study of centrosome aberrations in cancer cells.
While the image acquisition in such pipelines is automated, manual detectionof centrioles is still necessary to 
select cells for re-imaging at higher magnification.




In this repository we propose an algorithm which performs
this step automatically and with high accuracy. From the image labels produced by human experts and a 3D model of 
a centriole we constructan additional training set with patch-level labels. 
A two-level DenseNetis trained on the hybrid training data with synthetic patches and realimages, 
achieving much better results on real patient data than trainingonly at the image-level.


## Publication

This work has been accepted to MICCAI 2019 and you can find a [preprint on arXiv](https://arxiv.org/abs/1908.10109).
