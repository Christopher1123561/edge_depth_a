# edge_depth_a
Depth Estimation with Edge Enhancement
This release supports joint training a depth prediction model and a motion prediction model using only pairs of RGB images. The depth model infers a dense depth map from a single image. The motion model infers a 6 degree-of-freedom camera motion and a 3D dense motion field for every pixel from a pair of images. The approach does not need any auxiliary semantic information from the images, and the camera intrinsics can be either specified or learned.

Sample command line:
python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=$MY_CHECKPOINT_DIR \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "$MY_DATA_DIR"
      }
    },
    "trainer": {
      "init_ckpt": "$MY_IMAGENET_CHECKPOINT",
      "init_ckpt_type": "imagenet"
    }
  }'
