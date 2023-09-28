###
"""A binary for training depth and egomotion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

import depth_motion_field_model
import training_utils


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  training_utils.train(depth_motion_field_model.input_fn,
                       depth_motion_field_model.loss_fn,
                       depth_motion_field_model.get_vars_to_restore_fn)


if __name__ == '__main__':
  app.run(main)
