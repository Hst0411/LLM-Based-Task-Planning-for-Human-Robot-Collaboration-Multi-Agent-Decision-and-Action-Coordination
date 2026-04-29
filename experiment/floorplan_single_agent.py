from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
import json
import os
"""
A single agent simulation in a floorplan scene.

The Replicant navigates to a container and picks it up.
"""

def record_position(rep, positions_dict, frame_count):
    rep_pos = rep.dynamic.transform.position
    # 若 rep_pos 是 NumPy array 或 Vector3 類型
    image_name = f"img_{frame_count:04d}.jpg"
    positions_dict[image_name] = rep_pos.tolist()  # 安全轉成 list



def get_room_name(x, z):
    if -6 <= x < -2 and 4 <= z <= 8:
        return "Living Room"
    elif -2 <= x < 2 and 4 <= z <= 8:
        return "Dining Room"
    elif 2 <= x <= 6 and 4 <= z <= 8:
        return "Kitchen"
    elif -6 <= x < -2 and 0 <= z < 4:
        return "Bedroom 1"
    elif -2 <= x < 2 and 0 <= z < 4:
        return "Bedroom 2"
    elif 2 <= x <= 6 and 0 <= z < 4:
        return "Bedroom 3"
    elif -2 <= x < 0 and 6 <= z <= 8:
        return "Bathroom"
    elif 4 <= x <= 6 and 6 <= z <= 8:
        return "Study Room"
    else:
        return "Unknown Area"

def print_scene_object_container(c):
 
    # Print Containers
    print("=== Containers ===")
    for cid in c.state.container_ids:
        obj = c.object_manager.objects_static[cid]
        pos = c.object_manager.transforms[cid].position
        print(f"ID: {cid}, Name: {obj.name}, Position: {pos}")

    # Print Objects
    print("=== Target Objects ===")
    for tid in c.state.target_object_ids:
        obj = c.object_manager.objects_static[tid]
        pos = c.object_manager.transforms[tid].position
        print(f"ID: {tid}, Name: {obj.name}, Position: {pos}")




def senario():
    
    c = TransportChallenge(screen_width=1280, screen_height=720)

    human0_position = {"x": -3.8, "y": 0, "z": 3.46}
    human1_position = {"x": 1.5, "y": 0, "z": 0.0}
    c.start_floorplan_trial(scene="2a", layout=0, replicants=[human0_position],
                            num_containers=4, num_target_objects=8, random_seed=1)

    # ThirdPersonCamera
    camera = ThirdPersonCamera(avatar_id="camera",
                            position={"x": 0, "y": 10, "z": 0},
                            look_at=c.replicants[0].replicant_id)
    c.add_ons.append(camera)

    # FirstPersonCamera
    # firstcamera = FirstPersonAvatar(avatar_id="camera", position={"x": 1.5, "y": 0, "z": 0.0},
    #                                 camera_height=5.0, move_speed=10.0)
    # c.add_ons.append(firstcamera)

    c.communicate([{"$type": "set_floorplan_roof",
                    "show": False}])
    
    
    # ---------------------------------------------------------------
    # define image 儲存路徑，output path 為 image_capture 子目錄
    # print 儲存 image 的路徑
    # path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("robot_observation_data")
    print("------------------------------------------------")
    print(f"Images will be saved to: robot_observation_data")
    print("------------------------------------------------")


    # 建立一個 ImageCapture module, 用於擷取和保存影像:
    # avatar_ids=["test_camera"]: 指定要從攝影機 "test_camera" 擷取影像
    # path=path: 影像保存的路徑
    capture = ImageCapture(path="experiment/robot_observation_data/pickup_bowl", avatar_ids=["camera"])

    # 將 camera 和 capture 加入 Controller
    c.add_ons.extend([capture])
    # ---------------------------------------------------------------


    # print scene object
    print_scene_object_container(c)

    positions = {}
    frame_count = 0
    # human 0
    # if avoid == True, robot will try to "stop" before colliding with objects.
    c.replicants[0].collision_detection.avoid = False
    
    # human0 head to orange container
    c.replicants[0].navigate_to(target=c.state.container_ids[0])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        record_position(c.replicants[0], positions, frame_count)
        frame_count += 1
    
    c.communicate([])
    record_position(c.replicants[0], positions, frame_count)
    frame_count += 1
    print("Navigate status:", c.replicants[0].action.status)

    # human0 pick_up orange container
    c.replicants[0].pick_up(target=c.state.container_ids[0])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        record_position(c.replicants[0], positions, frame_count)
        frame_count += 1

    c.communicate([])
    record_position(c.replicants[0], positions, frame_count)
    frame_count += 1
    print("Pick up status:", c.replicants[0].action.status)

    # 保證最後一張圖的位置也被記錄
    c.communicate([])
    record_position(c.replicants[0], positions, frame_count)
    frame_count += 1

    # save agent position to agent_position.json
    output_dir = "experiment/robot_observation_data/pickup_bowl"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "agent_position.json"), "w") as f:
        json.dump(positions, f, indent=2)

    # human 1
    # c.replicants[1].collision_detection.avoid = False
    # c.replicants[1].navigate_to(target=c.state.container_ids[1])
    # while c.replicants[1].action.status == ActionStatus.ongoing:
    #     c.communicate([])
    
    # c.communicate([])
    # print(c.replicants[1].action.status)

    # c.replicants[1].pick_up(target=c.state.container_ids[1])
    # while c.replicants[1].action.status == ActionStatus.ongoing:
    #     c.communicate([])
    
    # c.communicate([])
    # print(c.replicants[1].action.status)
    # c.communicate({"$type": "terminate"})



if __name__ == "__main__":
    senario()
