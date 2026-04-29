from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.replicant.arm import Arm
import subprocess

def perception_analysis():
    # 呼叫 yolov8_gpt.py 執行分析
    subprocess.run(["python", "D:/Co-LLM-Agents-master/perception/yolov8_gpt.py"])

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


def sync_camera_to_replicant(c, replicant, height_offset=1.0, forward_distance=0.3, fov=90):
    """
    讓 camera avatar 移動到 replicant 頭部，並朝向他面對的方向，並調整 FOV。
    """
    rep_pos = replicant.dynamic.transform.position
    forward = replicant.dynamic.transform.forward

    cam_pos = {
        "x": float(rep_pos[0]),
        "y": float(rep_pos[1]) + height_offset,
        "z": float(rep_pos[2]) - 0.5
    }

    look_at = {
        "x": cam_pos["x"] + forward[0] * forward_distance,
        "y": cam_pos["y"] + forward[1] * forward_distance,
        "z": cam_pos["z"] + forward[2] * forward_distance
    }

    c.communicate([
        {
            "$type": "teleport_avatar_to",
            "position": cam_pos,
            "avatar_id": "a"
        },
        {
            "$type": "look_at_position",
            "position": look_at,
            "avatar_id": "a"
        },
        {
            "$type": "set_field_of_view",
            "field_of_view": fov,   # 加上這行：設定攝影機視野大小
            "avatar_id": "a"
        }
    ])

# Print scene object
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

    replicant_position = {"x": -3.8, "y": 0, "z": 3.46}
    c.start_floorplan_trial(scene="2a", layout=0, replicants=[replicant_position],
                            num_containers=4, num_target_objects=8, random_seed=1)

    # ThirdPersonCamera
    # camera = ThirdPersonCamera(avatar_id="camera",
    #                         position={"x": 0, "y": 35, "z": 0},
    #                         look_at={"x": 0, "y": 0, "z": 0})
    # c.add_ons.append(camera)

    # c.communicate([{"$type": "set_floorplan_roof", "show": False}])

    # 儲存路徑
    # print("------------------------------------------------")
    # print(f"Images will be saved to: experiment/robot_firstperson")
    # print("------------------------------------------------")

    # path=path: 影像保存的路徑
    # capture = ImageCapture(path="experiment/robot_observation_data/pickup_bowl", avatar_ids=["camera"])

    # # 將 camera 和 capture 加入 Controller
    # c.add_ons.extend([capture])

    # print scene object
    print_scene_object_container(c)

    # Disable collision detection
    c.replicants[0].collision_detection.avoid = False

    # # 初始同步 camera 位置
    # c.communicate([])  # 確保初始化
    # sync_camera_to_replicant(c, c.replicants[0])

    # navigate to food 8
    c.replicants[0].navigate_to(target=c.state.target_object_ids[8])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        # sync_camera_to_replicant(c, c.replicants[0])

    c.communicate([])
    print("Navigate status:", c.replicants[0].action.status)

    # pickup food 8
    c.replicants[0].pick_up(target=c.state.target_object_ids[8])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        # sync_camera_to_replicant(c, c.replicants[0])

    c.communicate([])
    print("Pick up status:", c.replicants[0].action.status)


    # # navigate to food 8
    # c.replicants[0].navigate_to({"x": 6.296, "y": 0, "z": 0.525})
    # while c.replicants[0].action.status == ActionStatus.ongoing:
    #     c.communicate([])
    #     # sync_camera_to_replicant(c, c.replicants[0])

    # c.communicate([])
    # print("Navigate status:", c.replicants[0].action.status)

    # # drop food 8
    # c.replicants[0].drop(arm=Arm.right)
    # while c.replicants[0].action.status == ActionStatus.ongoing:
    #     c.communicate([])
    #     # sync_camera_to_replicant(c, c.replicants[0])

    # c.communicate([])
    # print("Drop status:", c.replicants[0].action.status)

    # ================================================================
    # navigate to food 9
    c.replicants[0].navigate_to(target=c.state.target_object_ids[4])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        # print(c.replicants[0].action.status)
        # sync_camera_to_replicant(c, c.replicants[0])

    c.communicate([])
    print("Navigate status:", c.replicants[0].action.status)
    print(c.replicants[0].dynamic.transform.position)

    # pickup food 9
    c.replicants[0].pick_up(target=c.state.target_object_ids[4])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        # sync_camera_to_replicant(c, c.replicants[0])

    c.communicate([])
    print("Pick up status:", c.replicants[0].action.status)

    # # navigate to food 9
    # c.replicants[0].navigate_to({"x": -3.996, "y": 0, "z": -1.725})
    # while c.replicants[0].action.status == ActionStatus.ongoing:
    #     c.communicate([])
    #     # sync_camera_to_replicant(c, c.replicants[0])

    # c.communicate([])
    # print("Navigate status:", c.replicants[0].action.status)

    # # drop food 9
    # c.replicants[0].drop(arm=Arm.right)
    # while c.replicants[0].action.status == ActionStatus.ongoing:
    #     c.communicate([])
    #     # sync_camera_to_replicant(c, c.replicants[0])

    # c.communicate([])
    # print("Drop status:", c.replicants[0].action.status)
    for arm, obj_id in c.replicants[0].dynamic.held_objects.items():
        print(f"Arm: {arm.name} (type: {type(arm.name)}), Object ID: {obj_id} (type: {type(obj_id)})")


    print(c.replicants[0].dynamic.held_objects)



if __name__ == "__main__":
    senario()