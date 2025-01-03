## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import time
import pyrealsense2 as rs

# Import Numpy for easy array manipulation
import numpy as np

SOD = True

# Import OpenCV for easy image rendering
if SOD:
    from infer import infer
import cv2
from PIL import Image

import streamlit as st

# st.set_page_config(
#     page_title="S-MultiMAE",
#     page_icon="🧊",
#     layout="wide",
#     # initial_sidebar_state="expanded",
#     # menu_items={
#     #     'Get Help': 'https://www.extremelycoolapp.com/help',
#     #     'Report a bug': "https://www.extremelycoolapp.com/bug",
#     #     'About': "# This is a header. This is an *extremely* cool app!"
#     # }
# )

st.title("S-MultiMAE")
FRAME_WINDOW = st.image([])

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        original_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = np.array(
            original_depth_image / np.max(original_depth_image) * 256, dtype=np.uint8
        )
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.array(color_image / np.max(color_image) * 256, dtype=np.uint8)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        # print(
        #     "color_image",
        #     np.min(color_image),
        #     np.max(color_image),
        #     np.mean(color_image),
        #     color_image.dtype,
        # )

        if SOD:
            # start_time = time.time()
            sod_image = infer(
                image=Image.fromarray(color_image), depth=Image.fromarray(depth_image)
            )
            # print("Processing time", time.time() - start_time)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        # depth_image_3d = np.dstack(
        #     (depth_image, depth_image, depth_image)
        # )  # depth image is 1 channel, color is 3 channels
        # bg_removed = np.where(
        #     (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
        #     grey_color,
        #     color_image,
        # )

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(original_depth_image, alpha=0.03),
            cv2.COLORMAP_RAINBOW,
        )
        if SOD:
            images = np.hstack((color_image, depth_colormap, sod_image))
        else:
            images = np.hstack((color_image, depth_colormap))

        FRAME_WINDOW.image(Image.fromarray(images))
        if SOD:
            time.sleep(4)

        # cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
        # cv2.imshow("Align Example", images)
        # key = cv2.waitKey(1)
        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord("q") or key == 27:
        #     cv2.destroyAllWindows()
        #     break
finally:
    pipeline.stop()
