from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.challenge_state import ChallengeState
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH

# food
# camera = ThirdPersonCamera(position={"x": 0.7, "y": 0.7, "z": 0.7},
#                            avatar_id="a",
#                            look_at={"x": 0, "y": 0, "z": 0})
# path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("image_capture")
# print(f"Images will be saved to: {path}")
# capture = ImageCapture(avatar_ids=["a"], path=path)

# c = Controller(port=8889)
# c.add_ons.extend([camera, capture])
# object_id = c.get_unique_id()
# commands = [TDWUtils.create_empty_room(12, 12),
#             {"$type": "set_screen_size",
#              "width": 1024,
#              "height": 1024}]
# commands.extend(c.get_add_physics_object(model_name="f10_apple_iphone_4",
#                                          position={"x": 0, "y": 0, "z": 0},
#                                          object_id=object_id,
#                                          rotation={"x": 0, "y": 0, "z": 0},
#                                          scale_factor={"x": 3.5, "y": 3.5, "z": 3.5}))
# c.communicate(commands)
# c.communicate({"$type": "terminate"})

# replicant camera
c = Controller(port=8889)
camera = ThirdPersonCamera(position={"x": 1.5, "y": 2.2, "z": 1.5}, 
                           avatar_id="a", 
                           look_at={"x": -2, "y": -1, "z": -2})
c.add_ons.extend([camera])

state = ChallengeState()
commands = [TDWUtils.create_empty_room(12, 12),
            {"$type": "set_screen_size",
             "width": 1024,
             "height": 1024}]
replicant = ReplicantTransportChallenge(replicant_id=0,
                                        state=state,
                                        position={"x": 0, "y": 0, "z": 0},
                                        enable_collision_detection=False,
                                        rotation={"x": 0, "y": -125, "z": 0},
                                        name="replicant_0")
c.add_ons.extend([replicant])
c.communicate([{"$type": "set_screen_size",
             "width": 1024,
             "height": 1024}])
data = c.communicate([])
for i in range(len(data) - 1):
    r_id = OutputData.get_data_type_id(data[i])
    if r_id == 'imag':
        images = Images(data[i])
        if images.get_avatar_id() == "a":
            TDWUtils.save_images(images=images, filename= f"0", output_directory = 'experiment/detection_data/human/')


c.communicate(commands)
c.communicate({"$type": "terminate"})