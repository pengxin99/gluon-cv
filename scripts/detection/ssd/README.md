# Single Shot Multibox Object Detection [1]

[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)

- `--dali` Use [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) for faster data loading and data preprocessing in training with COCO dataset. DALI >= 0.12 required.
- `--amp` Use [Automatic Mixed Precision training](https://mxnet.incubator.apache.org/versions/master/tutorials/amp/amp_tutorial.html), automatically casting FP16 where safe.
- `--horovod` Use [Horovod](https://github.com/horovod/horovod) for distributed training, with a network agnostic wrapper for the optimizer, allowing efficient allreduce using OpemMPI and NCCL.

## Inference/Calibration Tutorial

### Float32 Inference

```
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=1
```

### Calibration

Naive calibrate model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=32 --calibration
```

### INT8 Inference

```
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=1 --deploy --model-prefix=./model/ssd_512_mobilenet1.0_voc-quantized-naive
```

## Performance

model | f32 latency(ms) | s8 latency(ms) | f32 throughput(fps, BS=256) | s8 throughput(fps, BS=256) | f32 accuracy | s8 accuracy
-- | -- | -- | -- | -- | -- | --
ssd_300_vgg16_atrous_voc | 105.60 | 13.08 | 19.47 | 110.14 | 77.49 | 77.49
ssd_512_vgg16_atrous_voc | 215.05 | 32.63 | 6.76 | 36.56 | 78.82 | 78.82
ssd_512_mobilenet1.0_voc | 28.98 | 6.97 | 65.55 | 210.17 | 75.51 | 75.49
ssd_512_resnet50_v1_voc | 52.77 | 11.75 | 28.68 | 143.61 | 80.24 | 80.23


## Auto-tuning with LPiT
```
usage: eval_ssd.py [-h] [--network NETWORK] [--deploy]
                   [--model-prefix MODEL_PREFIX] [--quantized]
                   [--data-shape DATA_SHAPE] [--batch-size BATCH_SIZE]
                   [--benchmark] [--num-iterations NUM_ITERATIONS]
                   [--dataset DATASET] [--num-workers NUM_WORKERS]
                   [--num-gpus NUM_GPUS] [--pretrained PRETRAINED]
                   [--save-prefix SAVE_PREFIX] [--calibration]
                   [--num-calib-batches NUM_CALIB_BATCHES]
                   [--quantized-dtype {auto,int8,uint8}]
                   [--calib-mode CALIB_MODE] [--auto_tuning]
                   [--auto_tuning_config AUTO_TUNING_CONFIG]

Eval SSD networks.

optional arguments:
  -h, --help            show this help message and exit
  --network NETWORK     Base network name
  --deploy              whether load static model for deployment
  --model-prefix MODEL_PREFIX
                        load static model as hybridblock.
  --quantized           use int8 pretrained model
  --data-shape DATA_SHAPE
                        Input data shape
  --batch-size BATCH_SIZE
                        eval mini-batch size
  --benchmark           run dummy-data based benchmarking
  --num-iterations NUM_ITERATIONS
                        number of benchmarking iterations.
  --dataset DATASET     eval dataset.
  --num-workers NUM_WORKERS, -j NUM_WORKERS
                        Number of data workers
  --num-gpus NUM_GPUS   number of gpus to use.
  --pretrained PRETRAINED
                        Load weights from previously saved parameters.
  --save-prefix SAVE_PREFIX
                        Saving parameter prefix
  --calibration         quantize model
  --num-calib-batches NUM_CALIB_BATCHES
                        number of batches for calibration
  --quantized-dtype {auto,int8,uint8}
                        quantization destination data type for input data
  --calib-mode CALIB_MODE
                        calibration mode used for generating calibration table
                        for the quantized symbol; supports 1. none: no
                        calibration will be used. The thresholds for
                        quantization will be calculated on the fly. This will
                        result in inference speed slowdown and loss of
                        accuracy in general. 2. naive: simply take min and max
                        values of layer outputs as thresholds for
                        quantization. In general, the inference accuracy
                        worsens with more examples used in calibration. It is
                        recommended to use `entropy` mode as it produces more
                        accurate inference results. 3. entropy: calculate KL
                        divergence of the fp32 output and quantized output for
                        optimal thresholds. This mode is expected to produce
                        the best inference accuracy of all three kinds of
                        quantized models if the calibration dataset is
                        representative enough of the inference dataset.
  --auto_tuning         Use with --only_inference, If set, will use auto-
                        tuning tool to do INT8 inference.
  --auto_tuning_config AUTO_TUNING_CONFIG
                        Config for auto-tuning tool. Must provide when use
                        --auto_tuning. default is ./config_ssd.ini



python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=256 --auto_tuning
```


## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
