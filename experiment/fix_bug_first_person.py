from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.replicant.arm import Arm
import numpy as np
import shutil, os, glob
from experiment.RHP.food_3d_detector import detect_visible_foods

def sync_camera_to_replicant(c, replicant, avatar_id="a",  height_offset=1.88, forward_distance=0.3, fov=90):
    """
    讓 camera avatar 移動到 replicant 頭部，並朝向他面對的方向，並調整 FOV。
    """
    rep_pos = replicant.dynamic.transform.position
    forward = replicant.dynamic.transform.forward

    cam_pos = {
        "x": float(rep_pos[0]),
        "y": float(rep_pos[1]) + height_offset,
        "z": float(rep_pos[2])
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
            "avatar_id": avatar_id
        },
        {
            "$type": "look_at_position",
            "position": look_at,
            "avatar_id": avatar_id
        },
        {
            "$type": "set_field_of_view",
            "field_of_view": fov,   # 加上這行：設定攝影機視野大小
            "avatar_id": avatar_id
        }
    ])

# Print scene object
def print_scene_object_container(c):

    # Print Containers
    print("=== Containers ===")
    for cid in c.state.container_ids:
        obj = c.object_manager.objects_static[cid]
        pos = c.object_manager.transforms[cid].position
        # 假設 pos 是 numpy.ndarray 類型，並四捨五入到小數點後四位
        pos = np.round(pos, 4)  # 對整個數組進行四捨五入

        # 使用 f-string 格式化輸出，保證顯示四位小數
        print(f"ID: {cid}, Name: {obj.name}, Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # Print Objects
    print("=== Target Objects ===")
    i = 0
    for tid in c.state.target_object_ids:
        obj = c.object_manager.objects_static[tid]
        pos = c.object_manager.transforms[tid].position
        # 假設 pos 是 numpy.ndarray 類型，並四捨五入到小數點後四位
        pos = np.round(pos, 4)  # 對整個數組進行四捨五入

        # 使用 f-string 格式化輸出，保證顯示四位小數
        print(f"{i}, ID: {tid}, Name: {obj.name}, Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        i+=1
    
    print("======================================================================")

def senario():
    c = TransportChallenge(screen_width=640, screen_height=360)

    replicant_position = {"x": -3.8, "y": 0, "z": 3.46}
    replicant2_position = {"x": 4.0, "y": 0, "z": 2.72}
    c.start_floorplan_trial(scene="2a", layout=0, replicants=[replicant2_position],
                            num_containers=4, num_target_objects=8, random_seed=1)

    # ThirdPersonCamera
    # camera = ThirdPersonCamera(avatar_id="camera",
    #                         position={"x": 0, "y": 35, "z": 0},
    #                         look_at={"x": 0, "y": 0, "z": 0})
    # c.add_ons.append(camera)
    
    # c.communicate([
    #     {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "r0"}
    #     # {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "r2"}
    # ])

    # path=path: 影像保存的路徑
    output_path = "experiment/robot_observation_data/pickup_bowl"
    # capture = ImageCapture(path="experiment/robot_observation_data/pickup_bowl", avatar_ids=["r0", "r2"], pass_masks=["_img", "_depth"])
    capture = ImageCapture(path="experiment/robot_observation_data/pickup_bowl", avatar_ids=["a"], pass_masks=["_img", "_depth"])
    # capture = ImageCapture(path="experiment/robot_observation_data/pickup_bowl", avatar_ids=["a"])
    # 將 camera 和 capture 加入 Controller
    c.add_ons.append(capture)

    # warmup: 執行一個 frame，確保 ImageCapture 建立並有第一張影像可用
    c.communicate([])

    # 找出資料夾內所有 .jpg 檔案
    image_folder1 = "experiment/robot_observation_data/pickup_bowl/a"
    image_files = glob.glob(os.path.join(image_folder1, "*.jpg"))
    image_files += glob.glob(os.path.join(image_folder1, "*.png"))
    
    # 刪除每一個 .jpg 檔案
    for f in image_files:
        os.remove(f)

    # image_folder2 = "experiment/robot_observation_data/pickup_bowl/r2"
    # image_files = glob.glob(os.path.join(image_folder2, "*.jpg"))
    # image_files += glob.glob(os.path.join(image_folder2, "*.png"))

    # # 刪除每一個 .jpg 檔案
    # for f in image_files:
    #     os.remove(f)

    print("Deleted ALL IMAGES!!!")

    c.communicate([{"$type": "set_floorplan_roof", "show": False}])

    # 儲存路徑
    print("------------------------------------------------")
    print(f"Images will be saved to: {output_path}/a")
    print("------------------------------------------------")


    # print scene object
    print_scene_object_container(c)

    # Disable collision detection
    c.replicants[0].collision_detection.avoid = False

    # # 初始同步 camera 位置
    c.communicate([])  # 確保初始化
    sync_camera_to_replicant(c, c.replicants[0])
    # sync_camera_to_replicant(c, c.replicants[1], "r2")

    # navigate to food 4
    c.replicants[0].navigate_to(target=c.state.target_object_ids[4])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        sync_camera_to_replicant(c, c.replicants[0])
        # sync_camera_to_replicant(c, c.replicants[1], "r2")
        visible_foods = detect_visible_foods(controller=c,
                                            avatar_id="a",
                                            image_capture=capture,
                                            task_list_path="experiment/robot_task_list.json")
        # if visible_foods:
        #     print("👀 看到了以下食物：", visible_foods)


    c.communicate([])
    sync_camera_to_replicant(c, c.replicants[0])
    # sync_camera_to_replicant(c, c.replicants[1], "r2")
    print("Navigate status:", c.replicants[0].action.status)

    # pickup food 4
    c.replicants[0].pick_up(target=c.state.target_object_ids[4])
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        sync_camera_to_replicant(c, c.replicants[0])
        # sync_camera_to_replicant(c, c.replicants[1], "r2")

    c.communicate([])
    sync_camera_to_replicant(c, c.replicants[0])
    # sync_camera_to_replicant(c, c.replicants[1], "r2")
    print("Pick up status:", c.replicants[0].action.status)

    # navigate to food 4
    c.replicants[0].navigate_to({"x": -10.196, "y": 0, "z": 2.625})
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        sync_camera_to_replicant(c, c.replicants[0])
        # sync_camera_to_replicant(c, c.replicants[1], "r2")

    c.communicate([])
    sync_camera_to_replicant(c, c.replicants[0])
    # sync_camera_to_replicant(c, c.replicants[1], "r2")
    print("Navigate status:", c.replicants[0].action.status)

    # drop food 4
    c.replicants[0].drop(arm=Arm.right)
    while c.replicants[0].action.status == ActionStatus.ongoing:
        c.communicate([])
        sync_camera_to_replicant(c, c.replicants[0])
        # sync_camera_to_replicant(c, c.replicants[1], "r2")

    c.communicate([])
    sync_camera_to_replicant(c, c.replicants[0])
    # sync_camera_to_replicant(c, c.replicants[1], "r2")
    print("Drop status:", c.replicants[0].action.status)

    c.communicate({"$type": "terminate"})


if __name__ == "__main__":
    shutil.copyfile('experiment/task_list.json', 'experiment/robot_task_list.json')
    senario()
