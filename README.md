## Prerequisites

This code has been tested on Windows 10 and the following are the main components that need to be installed:

- environment set for torch cache home and huggingface cache home to pretrained-model folder on root path of project
- Python >= 3.9
- PyTorch (cpu or gpu base) right now code is purely running on cpu
- torchvision (cpu or gpu base) right now code is purely running on cpu
- requirements.txt

- download models from drive ( https://drive.google.com/drive/folders/1q2sR1mbqLhgoMnxjusCBDDCU_1CqXciX?usp=sharing )

  - put big-lama.pt inside pretrained-model/hub/checkpoints
  - put yolov8m-seg.pt inside pretrained-model/
  - put torch_model.p inside pretrained-model/

- maybe some libraries not found during first time run code. Please add wisely with pip
- used lama model for inpaiting, yolov8 for object detection

## Run Current Working code on image
```bash
python full-stack-server.py
```

- Run full-stack-server.py and you need to access web local url in browser for testing


## Train the model

```bash
python train.py --config configs/config.yaml
```

The checkpoints and logs will be saved to `checkpoints`。

## Test with the trained model

By default, it will load the latest saved model in the checkpoints. You can also use `--iter` to choose the saved models by iteration.

```bash
python test_single.py \
	--image examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png \
	--mask examples/center_mask_256.png \
	--output examples/output.png
```

## Test with the converted TF model:

```bash
python test_tf_model.py \
	--image examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png \
	--mask examples/center_mask_256.png \
	--output examples/output.png \
	--model-path torch_model.p
```
