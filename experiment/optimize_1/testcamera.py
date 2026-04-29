from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH

c = Controller(port=8889)

# Add two cameras.
cam_0 = ThirdPersonCamera(avatar_id="c0",
                          position={"x": 1, "y": 2.2, "z": -0.5},
                          rotation={"x": 0, "y": -45, "z": 0})
cam_1 = ThirdPersonCamera(avatar_id="c1",
                          position={"x": 2, "y": 1, "z": -5})

path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("image_capture")
print(f"Images will be saved to: {path}")

# Enable image capture for both cameras.
cap = ImageCapture(path=path, avatar_ids=["c0", "c1"])

c.add_ons.extend([cam_0, cam_1, cap])
command = [TDWUtils.create_empty_room(12, 12),
            {"$type": "set_screen_size",
            "width": 1024,
            "height": 800}]
c.communicate(command)

