# Docker images

There are a lot of components. Best way to deal with dependencies is to use docker.

```bash

# build
docker build -t weather-gan .

# run
docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace \
  --gpus device=0 \
  --shm-size 64G \
  weather-gan:latest
```

# Data prep

Data must be organized in this fashion.


```
feamgan
   data
      sunny
         zips
            data1.zip
            data2.zip
            ...
         sequencens
            train
               frames
                  sequence1
                     sequence1_frame1_info.png
                     sequence1_frame2_info.png
                     ...
                  sequence2
                     sequence2_frame1_info.png
                     sequence2_frame2_info.png
                     ...
                  ...
               segmentations
                  ...
               ...
            val
               ...
      cloudy
         zips
            data1.zip
            data2.zip
            ...
         sequencens
            train
               frames
                  sequence1
                     sequence1_frame1_info.png
                     sequence1_frame2_info.png
                     ...
                  sequence2
                     sequence2_frame1_info.png
                     sequence2_frame2_info.png
                     ...
                  ...
               segmentations
                  ...
               ...
            val
               ...
```

Segmentation is required on all data just during training phase. Download the ```mseg-3m.pth``` model [here](https://drive.google.com/file/d/1BeZt6QXLwVQJhOVd_NTnVTmtAO1zJYZ-/view), and add it to models.

```bash
python segment_batch.py --dataset_path /data/sunny
python segment_batch.py --dataset_path /data/cloudy
```

# Train

1) Set up weight and biases config. 

Edit `controllerConfig.json` (experiment tracking)
Edit `experimentSchedule.json` (experiment config file)

If running inference, it should be something like:

```json
{
  "mode": "sequential",
  "experimentsToRun": ["InferenceExperiment"],
  "experimentConfigsToRun": {"InferenceExperiment": [
    "src/weathergan/feamgan/Experiment_Component/ExperimentConfigs/pretrainedModels/val/FeaMGAN_Cloudy_to_Sunny_Crop352_Full.json"
    ]}
}
```

If training, it should be something like:

```json
{
   "mode": "sequential",
   "experimentsToRun": ["TrainModelsExperiment"],
   "experimentConfigsToRun": {"TrainModelsExperiment": ["src/weathergan/feamgan/Experiment_Component/ExperimentConfigs/train/FeaMGAN/cloudy_to_sunny/FeaMGAN_cloudy_to_sunny_Crop352_Full_r5.json"]}
}
```

Now to start training, 

```bash
python main.py --experiment_schedule_path ./experimentSchedule.json --controller_config_path ./controllerConfig.json
```