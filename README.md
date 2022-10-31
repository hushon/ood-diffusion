# OoD-Diffusion

## Environments

```
Python 3.8, PyTorch 1.10, CUDA 11.3
```

## I/O format
JSON format is used to store the inputs/outputs of this module.
Input includes the training images and the test images.
Ouput includes the per-sample OoD evaluation scores.

`sample_input.json`
```
{
    "train_images": [
        "./dataset/train/0.jpg",
        "./dataset/train/1.jpg",
        "./dataset/train/2.jpg",
        "./dataset/train/3.jpg",
        "./dataset/train/4.jpg"
    ],

    "test_images": [
        "./dataset/test/0.jpg",
        "./dataset/test/1.jpg"
    ]
}
```

`sample_output.json`
```
{
    "test_ood_score": {
        "./dataset/test/0.jpg": 0.1111,
        "./dataset/test/1.jpg": 0.7999
    }
}
```

## Training

```
python main.py --phase train --log_dir ./results/ --input_json_path ./sample_io/sample_input.json
```

The checkpoint file will be stored in the `log_dir` directory.


## Testing

```
python main.py --phase test --ckpt_path <checkpoint path> --input_json_path ./sample_io/sample_input.json --ouput_json_path ./sample_io/sample_output.json
```
The evaluation results will be stored to `ouput_json_path` file.