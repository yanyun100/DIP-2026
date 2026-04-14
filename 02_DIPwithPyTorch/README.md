This repository is Zhuowen Sun's implementation of Assignment_02 of DIP.

## Requirements
To install requirements:
```basic
python -m pip install -r requirements.txt
```
## Running
To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```
python run_blending_gradio.py
```
## Results

### Poission Image Editing

Bear in water:

![image](https://github.com/stargazing0987/DIP/blob/master/Assignment2/figures/BearInWater/result.png)

Equation in scene:

![image](https://github.com/stargazing0987/DIP/blob/master/Assignment2/figures/Equation/result.png)

Monalisa:

![image](https://github.com/stargazing0987/DIP/blob/master/Assignment2/figures/Monalisa/result.png)

Shark with guys:

![image](https://github.com/stargazing0987/DIP/blob/master/Assignment2/figures/Shark/result.png)

You can find the source images and target images in subfolder figures.

### Pix2Pix

Val_results after epoch 0,epoch 400 and epoch 795:

![image](https://github.com/stargazing0987/DIP/blob/master/Assignment2/Pix2Pix/figures/result.png)

You can find that after epoch 400, the loss doesn't decrease.And the results are not satisfying , that's because the dataset scale is so small to get good results.

