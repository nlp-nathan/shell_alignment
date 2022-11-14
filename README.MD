# Shell Alignment

Current [cropped images](https://drive.google.com/drive/folders/1dJkp5St1mhBfDSPrzfnWVfg8Tk11hPy7?usp=share_link) and [keypoint annotations](https://drive.google.com/file/d/1Wx---YJIGxQH_BRbo5hCZHHVN-MoNHOu/view?usp=share_link)

# Three Steps:
1. Shell Detection (Object detection)
2. Keypoint Detection ([notebook](notebooks/keypoint_detection.ipynb))
3. Affine Transformations ([notebook](notebooks/affine_transformations.ipynb))


# Keypoint Detection

<p align="center">
<img src="images/shells.gif" width="384">
</p>
<p align="center">
Keypoint detection on unseen individuals.
</p>

Currently uses MobileNetV2. Trained with images received on October 9th. The data set was split 70/20/10 for train, val, and test sets respectively. Train set consisted of 130 images of 24 individuals. Validation set contains 2 individuals not in any other set, as well as 1 image from each individual from the train set (total 37 images). Test set consists of 1 unseen individual and 14 images from individuals in the train set, (total 19 images).

Model | Input Size | Val MAE | Val MSE | Test MAE | Test MSE |
----- | ---------- | ----- | ----- |---- |---- |
MobileNetV2 | 224 | 4.5982 | 125.8690 | 1.8656 | 5.9582 |
