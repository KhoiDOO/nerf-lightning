# NerfLightning - Simple NERF Archive
Project Website: https://khoidoo.github.io/nerf-lightning/

## Setup
```
python3 -m venv .env
source .env/bin/activate
python -m pip install -U pip

pip install wheel
pip3 install torch torchvision torchaudio
python -m pip install lightning
pip install omegaconf
pip install opencv-python
```

## Training
All training can be written in a ```YAML``` file. An example is located at ```./configs/tinynerf.yaml```, shown below
```yaml
name: nerf
save_dir: runs
trial_dir: ${get_trial_dir:${save_dir}}
save_cfg_path: ${path_append:${trial_dir},config.yaml}
seed: 0

data:
  data_source: data/src
  train_path: ${path_append:${data.data_source},training_data.pkl}
  valid_path: ${path_append:${data.data_source},testing_data.pkl}
  batch_size: 1024
  shuffle: ${get_shuffle:${train.trainer.devices}}
  num_workers: 24

system_type: TinyNerf
system:
  model_type: NerfModel
  model:
    hidden_dim: 256
    embedding_dim_pos: 10
    embedding_dim_direction: 4
  optimizer:
    name: Adam
    args:
      lr: 5.e-4
  scheduler:
    name: MultiStepLR
    args:
      milestones: [2, 4, 8]
      gamma: 0.5
      last_epoch: -1
  loss: 
    name: mse
    args:
      reduction: mean
  args:
    near: 2
    far: 6
    bins_count: 192
    eval_height: 400
    eval_width: 400

train:
  trainer:
    devices: [0]
    max_epochs: 20
    check_val_every_n_epoch: 1
    enable_progress_bar: True
    accumulate_grad_batches: 1
    log_every_n_steps: 50
    default_root_dir: ${trial_dir}
  logger:
    names: [WandbLogger, CSVLogger]
    args:
      wandb:
        name: ${get_run_id:${trial_dir}}
        project: nerf
        save_dir: ${path_append:${trial_dir},wandb}
        id: ${get_run_id:${trial_dir}}
        anonymous: True
        log_model: all
      csv:
        name: csv
        save_dir: ${trial_dir}
  callback:
    names: [ModelCheckpoint]
    args:
      modelcp:
        monitor: train/psnr
        mode: max
```
Note that the repo supports all [Pytorch's optimizers](https://pytorch.org/docs/stable/optim.html) and [Pytorch's Schedulers](https://pytorch.org/docs/stable/optim.html#module-torch.optim.lr_scheduler), specifying the accurate name from Pytorch and place all available args into args dictionary. The repo also supports all [Lightning's Logger](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) and [Lightning's Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html). To start training using ```CLI```:
```
python main.py --config ./configs/tinynerf.yaml
```
