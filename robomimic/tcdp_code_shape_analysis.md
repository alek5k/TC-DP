Here I am breaking down how the conditioning occurs in the diffusion policy:

During training:
diffusion_policy.py - `train_on_batch`:

During inference:
diffusion_policy.py - `_get_action_trajectory`:


inputs["obs"].keys()
[a.shape for k, a in inputs['obs'].items()]

```python
'agentview_image',  torch.Size([16, 2, 3, 84, 84])
'robot0_eef_pos',   torch.Size([16, 2, 3])
'robot0_eef_quat', torch.Size([16, 2, 4])
'robot0_eye_in_hand_image',   torch.Size([16, 2, 3, 84, 84])
'robot0_gripper_qpos',  torch.Size([16, 2, 2])
'robot0_joint_vel',  torch.Size([16, 2, 7])
'robot0_gripper_qvel',  torch.Size([16, 2, 2])
'sinusoidal_progress_encoding', torch.Size([16, 2, 128])
'idleness', torch.Size([16, 2, 1])
```

During training:
self.nets["policy"]["obs_encoder"]

During inference:
nets["policy"]["obs_encoder"]

```python
ObservationGroupEncoder(
    group=obs
    ObservationEncoder(
        Key(
            name=agentview_image
            shape=[3, 84, 84]
            modality=rgb
            randomizer=ModuleList(
              (0): CropRandomizer(input_shape=[3, 84, 84], crop_size=[76, 76], num_crops=1)
            )
            net=VisualCore(
              input_shape=[3, 76, 76]
              output_shape=[64]
              backbone_net=ResNet18Conv(input_channel=3, input_coord_conv=False)
              pool_net=SpatialSoftmax(num_kp=32, temperature=1.0, noise=0.0)
            )
            sharing_from=None
        )
        Key(
            name=idleness
            shape=[1]
            modality=low_dim
            randomizer=ModuleList(
              (0): None
            )
            net=None
            sharing_from=None
        )
        Key(
            name=robot0_eef_pos
            shape=[3]
            modality=low_dim
            randomizer=ModuleList(
              (0): None
            )
            net=None
            sharing_from=None
        )
        Key(
            name=robot0_eef_quat
            shape=[4]
            modality=low_dim
            randomizer=ModuleList(
              (0): None
            )
            net=None
            sharing_from=None
        )
        Key(
            name=robot0_eye_in_hand_image
            shape=[3, 84, 84]
            modality=rgb
            randomizer=ModuleList(
              (0): CropRandomizer(input_shape=[3, 84, 84], crop_size=[76, 76], num_crops=1)
            )
            net=VisualCore(
              input_shape=[3, 76, 76]
              output_shape=[64]
              backbone_net=ResNet18Conv(input_channel=3, input_coord_conv=False)
              pool_net=SpatialSoftmax(num_kp=32, temperature=1.0, noise=0.0)
            )
            sharing_from=None
        )
        Key(
            name=robot0_gripper_qpos
            shape=[2]
            modality=low_dim
            randomizer=ModuleList(
              (0): None
            )
            net=None
            sharing_from=None
        )
        Key(
            name=sinusoidal_progress_encoding
            shape=[128]
            modality=low_dim
            randomizer=ModuleList(
              (0): None
            )
            net=None
            sharing_from=None
        )
        output_shape=[266]
    )
)
```

Reading the above that means:
agentview_image feature dim: 64
idleness: 1
eef_pos: 3
eef_quat: 4
eye_in_hand_image feature dim: 64
robot0_gripper_qpos: 2
sinusoidal_progress_encoding: 128

64 + 1 + 3 + 4 + 64 + 2 + 128 = 266


obs_features.shape
```python
# during training:
torch.Size([16, 2, 266])

# during inference:
torch.Size([1, 2, 266])
```

obs_cond.shape
```python
# during training:
torch.Size([16, 532])

# during inference:
torch.Size([1, 532])
```
