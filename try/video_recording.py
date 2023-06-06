"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
import os

import imageio
import numpy as np

import robosuite.macros as macros
from robosuite import make
from robosuite.renderers import load_renderer_config

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--camera", type=str, default=["frontview", "birdview"])
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    # initialize an environment with offscreen renderer
    env = make(
        args.environment,
        args.robots,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        # renderer="nvisii",
        # renderer_config=load_renderer_config("nvisii"),
        # camera_segmentations=None,
    )

    obs = env.reset()
    ndim = env.action_dim

    camera_num = len(args.camera)
    print(camera_num)
    os.mkdir("./images")
    os.mkdir("./images/frontview")
    os.mkdir("./images/birdview")

    # create a video writer with imageio
    writer_front = imageio.get_writer("./images/frontview.mp4", fps=50)
    writer_bird = imageio.get_writer("./images/birdview.mp4", fps=50) 

    frames = []
    for i in range(args.timesteps):

        # run a uniformly random agent
        action = 0.5 * np.random.randn(ndim)
        obs, reward, done, info = env.step(action)

        # dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame_front = obs["frontview_image"]
            frame_bird = obs["birdview_image"]
            # print(frame.shape)
            writer_front.append_data(frame_front)
            writer_bird.append_data(frame_bird)
            with open(f'./images/frontview/{i}.npy', 'wb') as f:
                np.save(f, frame_front)
                f.close
            with open(f'./images/birdview/{i}.npy', 'wb') as f:
                np.save(f, frame_bird)
                f.close
            print("Saving frame #{}".format(i))

        if done:
            break

# with open('test.npy', 'wb') as f:
#     np.save(f, np.array([1, 2]))
#     np.save(f, np.array([1, 3]))

    writer_front.close()
    writer_bird.close()
