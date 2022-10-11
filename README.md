# feature-guided-flow

Implementation of "Improved Image Generation of Normalizing Flow with Semantic Feature Guidance" (TBA) in Pytorch

## Requirements

- PyTorch 1.7.1
- CUDA 11.0

> git clone https://github.com/dajinstory/feature-guided-flow.git <br/>
> cd feature-guided-flow <br/>
> pip install requirements.txt <br/>

## Usage

### Preparing Dataset, Configs

For training, you need to prepare Dataset and meta file. Meta file for Celeba dataset are in data/{DATASET_NAME}/train_meta.csv. It only contains the file name of dataset.

Also you should edit config files. There are "*_path" named keys. Those keys contains root path of dataset, path of meta file and path to save log and checkpoint files.

### Training Model

You can train model from scratch,
> bash script/train/train_fgflow_64x64_celeba.sh <br/>

resume from pretrained checkpoints,
> bash script/resume/train_fgflow_64x64_celeba.sh <br/>

and finetune from pretrained weights
> bash script/finetune/train_fgflow_64x64_celeba.sh <br/>

### Demo

You can check the sampling result of the pretrained model by running src/demo.ipynb

If you want to utilize the FGFlow model for your personal research, you just need to refer to src/demo.ipynb, src/model/ and src/loss.

### Pretrained Checkpoints

I trained 64x64 models on Celeba dataset for ???? iterations. The model followed the setting from GLOW official paper. I got bpd(bits per dimension) about ??,  . I trained 64x64 model with 1 GPU, 16 mini-batch size on each GPU. 

|      HParam       |          FGFlow64x64V0          |
| ----------------- | ----------------------------- |
| input shape       | (64,64,3)                     |
| L                 | 4                             |
| K                 | 48                            |
| hidden_channels   | 512                           |
| flow_permutation  | invertible 1x1 conv           |
| flow_coupling     | affine                        |
| batch_size        | 64 on each GPU, with 1 GPUs   |
| y_condition       | false                         |

|     Model     |   Dataset   |                              Checkpoint                                     |          Note         |
| ------------- | ----------- | --------------------------------------------------------------------------- | --------------------- |
| FGFlow64x64V0   | CelebA      | [FGFlow64X64V0_CelebA](TBA)  | 64x64 CelebA Dataset   |
| FGFlow256x256V0 | CelebA      | TBA  | Official Setting      |

## Samples

![Sample from Celeba, 64x64](doc/sample_64x64_celeba.png)

Generated samples. Left to right: Results by GLOW and Ours. We can see more details (e.g., hair, expression) in our results than in the GLOW baseline.

![Sample interpolation, 64x64](doc/sample_64x64_interpolation.png)

Result of z space interpolation.


## Reference
https://github.com/dajinstory/glow-pytorch <br/>

