# 目前裝了兩台 camera(r0、r2) 給 robots，主畫面是第三人稱
# 計算兩 robots 花費的 frames
# 協作策略：human 隨機 1 ~ 6
# robots 完成原分配的工作 or 發現 human 打亂 ---> LLM
from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.image_capture import ImageCapture
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.arm import Arm
import subprocess
from experiment.optimize_1.LLM_task_planner import get_two_robot_tasks_from_llm
from experiment.optimize_1.robot_idle_replanner import get_global_plan_from_llm
from experiment.optimize_1.human_finished_replanner import get_human_finished_tasks
from experiment.optimize_1.LLM_replan_anytime import report_progress_to_LLM
from experiment.robot_pereception_module import detect_and_check
from experiment.robot2_preception_module import detect_and_check2
import json
import numpy as np
import random
from typing import Union
import shutil, os, glob


def transform_to_dict(replicant_position, replicant2_position):
    replicant_position_dict = {
        'x': replicant_position[0],
        'y': replicant_position[1],
        'z': replicant_position[2]
    }
    replicant2_position_dict = {
        'x': replicant2_position[0],
        'y': replicant2_position[1],
        'z': replicant2_position[2]
    }
    return replicant_position_dict, replicant2_position_dict

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

def sync_camera_to_replicant_cmd(replicant, avatar_id="a", height_offset=1.88, forward_distance=0.3, fov=90):
    rep_pos = replicant.dynamic.transform.position
    forward = replicant.dynamic.transform.forward
    cam_pos = {"x": float(rep_pos[0]), "y": float(rep_pos[1]) + height_offset, "z": float(rep_pos[2])}
    look_at = {"x": cam_pos["x"] + forward[0] * forward_distance,
               "y": cam_pos["y"] + forward[1] * forward_distance,
               "z": cam_pos["z"] + forward[2] * forward_distance}
    return [
        {"$type": "teleport_avatar_to", "position": cam_pos, "avatar_id": avatar_id},
        {"$type": "look_at_position", "position": look_at, "avatar_id": avatar_id},
        {"$type": "set_field_of_view", "field_of_view": fov, "avatar_id": avatar_id}
    ]



# robot1 camera
# def sync_camera_to_replicant(c, replicant, height_offset=1.88, forward_distance=0.3, fov=90):
#     """
#     讓 camera avatar 移動到 replicant 頭部，並朝向他面對的方向，並調整 FOV。
#     """
#     rep_pos = replicant.dynamic.transform.position
#     forward = replicant.dynamic.transform.forward

#     cam_pos = {
#         "x": float(rep_pos[0]),
#         "y": float(rep_pos[1]) + height_offset,
#         "z": float(rep_pos[2])
#     }

#     look_at = {
#         "x": cam_pos["x"] + forward[0] * forward_distance,
#         "y": cam_pos["y"] + forward[1] * forward_distance,
#         "z": cam_pos["z"] + forward[2] * forward_distance
#     }

#     c.communicate([
#         {
#             "$type": "teleport_avatar_to",
#             "position": cam_pos,
#             "avatar_id": "r0"
#         },
#         {
#             "$type": "look_at_position",
#             "position": look_at,
#             "avatar_id": "r0"
#         },
#         {
#             "$type": "set_field_of_view",
#             "field_of_view": fov,   # 加上這行：設定攝影機視野大小
#             "avatar_id": "r0"
#         }
#     ])

# robot2 camera
# def sync_camera_to_replicant2(c, replicant, height_offset=1.88, forward_distance=0.3, fov=90):
#     """
#     讓 camera avatar 移動到 replicant 頭部，並朝向他面對的方向，並調整 FOV。
#     """
#     rep_pos = replicant.dynamic.transform.position
#     forward = replicant.dynamic.transform.forward

#     cam_pos = {
#         "x": float(rep_pos[0]),
#         "y": float(rep_pos[1]) + height_offset,
#         "z": float(rep_pos[2])
#     }

#     look_at = {
#         "x": cam_pos["x"] + forward[0] * forward_distance,
#         "y": cam_pos["y"] + forward[1] * forward_distance,
#         "z": cam_pos["z"] + forward[2] * forward_distance
#     }

#     c.communicate([
#         {
#             "$type": "teleport_avatar_to",
#             "position": cam_pos,
#             "avatar_id": "r2"
#         },
#         {
#             "$type": "look_at_position",
#             "position": look_at,
#             "avatar_id": "r2"
#         },
#         {
#             "$type": "set_field_of_view",
#             "field_of_view": fov,   # 加上這行：設定攝影機視野大小
#             "avatar_id": "r2"
#         }
#     ])

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

def mark_task_finished(food_id: Union[int, str], task_list_path="experiment/optimize_1/robot_task_list.json"):
    with open(task_list_path, "r") as f:
        task_list = json.load(f)
    
    updated = False
    for item in task_list:
        if (isinstance(food_id, int) and item["id"] == food_id) or \
           (isinstance(food_id, str) and item["name"] == food_id):
            item["state"] = "finished"
            updated = True
            break

    if updated:
        with open(task_list_path, "w") as f:
            json.dump(task_list, f, separators=(',', ':'))
        print(f"✅ Marked {food_id} as finished in {task_list_path}")
    else:
        print(f"🚫 Warning: food_id {food_id} not found in {task_list_path}")

# mark robot task_list both
def mark_task_finished_both(food_id: int):
    mark_task_finished(food_id, "experiment/optimize_1/robot_task_list.json")
    mark_task_finished(food_id, "experiment/optimize_1/robot2_task_list.json")

def is_close(pos1, pos2, threshold=0.03):
    a = np.array([pos1["x"], pos1["y"], pos1["z"]])
    b = np.array([pos2["x"], pos2["y"], pos2["z"]])
    return np.linalg.norm(a - b) < threshold

# reset task state
def reset_robot_task_list(task_list_path="experiment/optimize_1/robot_task_list.json"):
    # 讀取 JSON 檔案
    with open(task_list_path, 'r') as file:
        task_list = json.load(file)

    # 更新 task_list 中所有項目的 state 為 "unfinished"
    for task in task_list:
        task["state"] = "unfinished"

    # 儲存更新後的 task_list 回檔案
    with open(task_list_path, 'w') as file:
        json.dump(task_list, file, indent=4)

# ===== 模組化任務：navigate_to、pick_up、navigate_to、drop =====
def issue_action(replicant, action_name, index, target=None):
    # set agents task list path
    is_robot = index == 0
    is_robot2 = index == 2
    if is_robot:
        agent_name = "🤖 Robot1"
        task_path = "experiment/optimize_1/robot_task_list.json"
    elif is_robot2:
        agent_name = "🦾 Robot2"
        task_path = "experiment/optimize_1/robot2_task_list.json"
    else:
        agent_name = "👋 Human"
        task_path = "experiment/optimize_1/human_task_list.json"

    # if action_name in ["navigate_to", "pick_up"]:
    #     with open(task_path, "r") as f:
    #         current_tasks = json.load(f)
    #     finished_ids = [t["id"] for t in current_tasks if t["state"] == "finished"]

    #     # 若該食物已被完成，跳過此整組任務
    #     if target in finished_ids:
    #         print(f"🚫 Skipping action {action_name}: food {target} already finished.")
    #         return False

    if action_name == "navigate_to":
        if target is None:
            print("\u26a0\ufe0f Warning: navigate_to called with None target.")
            return False
        if isinstance(target, dict) and all(k in target for k in ("x", "y", "z")):
            # current_pos = {
            #     "x": replicant.dynamic.transform.position[0],
            #     "y": replicant.dynamic.transform.position[1],
            #     "z": replicant.dynamic.transform.position[2]
            # }
            # distance = np.linalg.norm(np.array([target["x"], target["y"], target["z"]]) -
            #                           np.array([current_pos["x"], current_pos["y"], current_pos["z"]]))
            # if distance < 0.15:
            #     print(f"🚫 {agent_name} skips navigation: already close to target {target}")

            #     # 嘗試標記該目標為 finished（回推該任務的 food_id）
            #     with open(task_path, "r") as f:
            #         task_list = json.load(f)
            #     for task in task_list:
            #         if is_close(task["target"], target):

            #             if index in [0, 2]:  # robot1 or robot2
            #                 mark_task_finished_both(task["id"])
            #             else:
            #                 mark_task_finished(task["id"], task_path)

            #             # mark_task_finished(task["id"], task_path)
            #             break

            #     return False
            # else:
            replicant.navigate_to(target)
        else:
            print(f"\u26a0\ufe0f Warning: Invalid navigation target: {target}")
            return False

    elif action_name == "pick_up":

        # 立即在 pick_up 完成後，標記任務為完成
        with open(task_path, "r") as f:
            task_list = json.load(f)

        # 找出剛剛 pick_up 的食物 ID 並馬上回報已拿取食物
        for task in task_list:
            if task["id"] == target:
                food_id = task["id"]
                print(f"✅ {food_id} picked up by {agent_name}. Marking as finished.")
                if index in [0, 2]:  # robot1 or robot2
                    mark_task_finished_both(food_id)
                else:
                    mark_task_finished(food_id, task_path)
                break

        replicant.pick_up(target)
    elif action_name == "drop":
        replicant.drop(arm=Arm.right)

    return True


# ---------------------
# robot task lists(LLM)
# ---------------------
def LLM_robot_tasks(c):
    return [
        ("navigate_to", {'x': -1.6637, 'y': 0, 'z': 2.6118}),
        ("pick_up", 6871504),
        ("navigate_to", {'x': 8.796, 'y': 0, 'z': 1.925}),
        ("drop", None),
        ("navigate_to", {'x': 6.246345, 'y': 0, 'z': 1.824576}),
        ("pick_up", 13875562),
        ("navigate_to", {'x': 6.296, 'y': 0, 'z': 0.525}),
        ("drop", None),
        ("navigate_to", {'x': -1.6217177549247004, 'y': 0, 'z': -3.863255628194672}),
        ("pick_up", 10724664),
        ("navigate_to", {'x': 6.0, 'y': 0, 'z': -1.4}),
        ("drop", None),
        ("navigate_to", {'x': 5.054031081339902, 'y': 0, 'z': -3.5}),
        ("pick_up", 1051562),
        ("navigate_to", {'x': -3.51, 'y': 0, 'z': 2.46319}),
        ("drop", None),
        ("navigate_to", {'x': 2.7463560104370117, 'y': 0, 'z': -2.4254207611083984}),
        ("pick_up", 14540030),
        ("navigate_to", {'x': -3.996, 'y': 0, 'z': -1.725}),
        ("drop", None),
        ("navigate_to", {'x': -8.187086722199306, 'y': 0, 'z': -1.262758481111447}),
        ("pick_up", 11530259),
        ("navigate_to", {'x': -10.196, 'y': 0, 'z': 2.625}),
        ("drop", None)
    ]

# ---------------------
# human task lists(手動)
# ---------------------
def get_human_tasks(task_list_path="experiment/optimize_1/human_task_list.json"):
    with open(task_list_path, "r") as f:
        tasks = json.load(f)
    task_sequence = []
    for task in tasks:
        task_sequence.extend([
            ("navigate_to", task["position"]),
            ("pick_up", task["id"]),
            ("navigate_to", task["target"]),
            ("drop", None)
        ])

    print("🙋 Human task sequence:")
    for i, task in enumerate(tasks):
        print(f"{i+1}. {task['id']} → from {task['position']} to {task['target']}")
    
    print("======================================================================")
    return task_sequence

def generate_human_task_list(all_tasks_path="experiment/optimize_1/task_list.json",
                              output_path="experiment/optimize_1/human_task_list.json"): 
    num_items = random.randint(1, 6)  # ✅ 隨機選 1~6 個
    # num_items = 6

    with open(all_tasks_path, 'r') as f:
        all_tasks = json.load(f)

    selected = random.sample(all_tasks, num_items)

    with open(output_path, 'w') as f:
        json.dump(selected, f, indent=4)
    
    print("======================================================================")
    print(f"✅ Human randomly selected {num_items} tasks.")

# 顯示 robot_task_list.json 中完成的 food ID
def print_finished_foods(task_list_path="experiment/optimize_1/robot_task_list.json"):
    with open(task_list_path, "r") as f:
        task_list = json.load(f)
    
    finished = [item["id"] for item in task_list if item["state"] == "finished"]

    print(f"✅ Finished food items in {task_list_path}:")
    for i, food_id in enumerate(finished):
        print(f"{i+1}. {food_id}")
    print(f"🎯 Total delivered: {len(finished)} / {len(task_list)}")
    print("======================================================================")

def senario():
    c = TransportChallenge(screen_width=640, screen_height=360)
    replicant_position = {"x": -3.8, "y": 0, "z": 3.46}
    replicant2_position = {"x": 4.0, "y": 0, "z": 2.82}
    human_position = {"x": -8.5, "y": 0, "z": 2.0}
    c.start_floorplan_trial(scene="2a", layout=0, replicants=[replicant_position, human_position, replicant2_position],
                            num_containers=0, num_target_objects=8, random_seed=1)
    
    # 🔥 加這段，確保 replicants 全部 spawn 完
    for i in range(5):
        resp = c.communicate([])
        print(f"warmup frame {i+1} ok")
        
    # 🔥 必須明確建立兩個 avatar
    c.communicate([
        {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "r0"},
        {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "r2"}
    ])
    
    # 找出資料夾內所有 .jpg 檔案
    image_folder1 = "experiment/multiple_camera/r0"
    image_folder2 = "experiment/multiple_camera/r2"

    # # 刪除每一個 .jpg 檔案
    for f in glob.glob(os.path.join(image_folder1, "*.jpg")):
        os.remove(f)
    for f in glob.glob(os.path.join(image_folder2, "*.jpg")):
        os.remove(f)

    # TDW 只能加入一個 image capture 模組，預設是顯示 avatar "a" 的相機畫面
    # 分別儲存兩 camera 拍到的畫面
    capture = ImageCapture(path="experiment/multiple_camera", avatar_ids=["r0", "r2"])
    c.add_ons.append(capture)

    c.communicate([{"$type": "set_floorplan_roof", "show": False}])

    # 儲存路徑
    print("------------------------------------------------")
    print(f"Images will be saved to: experiment/multiple_camera/r0")
    print(f"Images will be saved to: experiment/multiple_camera/r2")
    print("------------------------------------------------")

    # print scene object
    print_scene_object_container(c)

    # robot Disable collision detection
    c.replicants[0].collision_detection.avoid = False
    # human Disable collision detection
    c.replicants[1].collision_detection.avoid = False
    # robot Disable collision detection
    c.replicants[2].collision_detection.avoid = False
    
    print("GPT-4 開始規劃兩 robots 初始運輸順序")
    robot_tasks, robot2_tasks, previous_plan = get_two_robot_tasks_from_llm(
        task_list_path="experiment/optimize_1/task_list.json",
        robot1_position=replicant_position,
        robot2_position=replicant2_position
    )
    # robot_tasks = []
    # robot2_tasks = LLM_robot_tasks(c)
    # previous_plan = []

    # print(robot_tasks)
    # print(robot2_tasks)
    # print(previous_plan)

    # 隨機產生 human task
    generate_human_task_list()

    # human_tasks 手動設計
    human_tasks = get_human_tasks()
    # human_tasks = []

    indices = [0, 0, 0]
    active = [True, True, True]
    r0_frame_count = 0
    r2_frame_count = 0
    frame_count = 0
    human_interference = False
    r0_last_idx = 0
    r2_last_idx = 0

    while any(active):
        # 每一次 communicate 就代表 TDW 跑了一幀
        commands = []

        # camera 更新（只產生 command，不直接 call communicate）
        if active[0]:
            commands.extend(sync_camera_to_replicant_cmd(c.replicants[0], "r0"))
        if active[2]:
            commands.extend(sync_camera_to_replicant_cmd(c.replicants[2], "r2"))

        c.communicate(commands)
        # ✅ 只在 robot1 或 robot2 還在動的時候，才算 frame
        if active[0] or active[2]:
            frame_count += 1

        # Perception module
        if active[0]:
            r0_frame_count += 1
            if r0_frame_count % 100 == 0:
                r0_last_idx, events = detect_and_check(start_index=r0_last_idx)
                for e in events:
                    print("🔔 Event:", e)

                    # 偵測到 human 運送 food(mark_task_list(food))，robot 跳下一個工作
                    if "detected a human carrying" in e:
                        food_name = e.split("detected a human carrying")[-1].strip()
                        print(f"🍎 Robot1 detected human carrying food: {food_name}")
                        mark_task_finished_both(food_id=food_name)
                        # indices[0] += 4

        if active[2]:
            r2_frame_count += 1  
            if r2_frame_count % 100 == 0:
                r2_last_idx, events = detect_and_check2(start_index=r2_last_idx)
                for e in events:
                    print("🔔 Event:", e)

                    # 偵測到 human 運送 food(mark_task_list(food))，robot 跳下一個工作
                    if "detected a human carrying" in e:
                        food_name = e.split("detected a human carrying")[-1].strip()
                        print(f"🍎 Robot2 detected human carrying food: {food_name}")
                        mark_task_finished_both(food_id=food_name)
                        # indices[2] += 4
        
        for i in [0, 1, 2]:

            # human interference
            if human_interference:
                human_interference = False
                replicant_position = c.replicants[0].dynamic.transform.position
                replicant2_position = c.replicants[2].dynamic.transform.position
                replicant_position, replicant2_position = transform_to_dict(replicant_position, 
                                                                            replicant2_position)
                new_robot_tasks, new_robot2_tasks, previous_plan = get_human_finished_tasks(replicant_position, 
                                                                                            replicant2_position, 
                                                                                            history=previous_plan)
                
                # robot1 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects
                # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
                
                # robot1 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                if robot_hold_object:
                    print("robot1 is holding objects")
                    robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
                # 沒拿就直接把 new_robot_tasks 全部覆蓋
                else:
                    print("robot1 isn't holding objects")
                    robot_tasks = new_robot_tasks
                    indices[0] = 0
                
                # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                if robot2_hold_object:
                    print("robot2 is holding objects")
                    robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
                # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                else:
                    print("robot2 isn't holding objects")
                    robot2_tasks = new_robot2_tasks
                    indices[2] = 0
                
            # 🔥 robot1 完成，robot2 還有工作
            if indices[0] >= len(robot_tasks) and active[0]:
                print("🤖 Robot1 finished all tasks → re-planning with LLM")
                replicant_position = c.replicants[0].dynamic.transform.position
                replicant2_position = c.replicants[2].dynamic.transform.position
                replicant_position, replicant2_position = transform_to_dict(replicant_position, replicant2_position)
                new_robot_tasks, new_robot2_tasks, previous_plan = get_global_plan_from_llm(replicant_position, replicant2_position,
                                                                             history=previous_plan)                                
                # robot1 直接用新工作覆蓋，並從頭執行 (indices[0] = 0)
                if new_robot_tasks:
                    robot_tasks = new_robot_tasks
                    indices[0] = 0
                
                if new_robot2_tasks:
                    # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                    robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
                    if robot2_hold_object:
                        robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
                    # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                    else:
                        robot2_tasks = new_robot2_tasks
                        indices[2] = 0
                
                # robot2 繼續工作，關掉 robot1
                if not new_robot_tasks:
                    active[0] = False
                    print("🤖 Robot1 finished all task!")
                    break
                
                if not new_robot_tasks and not new_robot2_tasks:
                    active[0] = False
                    print("🤖 Robot1 finished all task!")
                    break

            # 🔥 robot2 完成，robot1 還有工作
            if indices[2] >= len(robot2_tasks) and active[2]:
                print("🦾 Robot2 finished all tasks → re-planning with LLM")
                replicant_position = c.replicants[0].dynamic.transform.position
                replicant2_position = c.replicants[2].dynamic.transform.position
                replicant_position, replicant2_position = transform_to_dict(replicant_position, replicant2_position)
                new_robot_tasks, new_robot2_tasks, previous_plan = get_global_plan_from_llm(replicant_position, replicant2_position,
                                                                             history=previous_plan)
                # robot2 直接用新工作覆蓋，並從頭執行 (indices[2] = 0)
                if new_robot2_tasks:
                    robot2_tasks = new_robot2_tasks
                    indices[2] = 0
                
                if new_robot_tasks:
                    # robot1 拿了東西 (第二個 navigate_to)，new_robot_tasks 加在下個工作
                    robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects
                    if robot_hold_object:
                        robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
                    # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                    else:
                        robot_tasks = new_robot_tasks
                        indices[0] = 0

                # robot1 繼續工作，關掉 robot2
                if not new_robot2_tasks:
                    active[2] = False
                    print("🦾 Robot2 finished all task!")
                    break
                
                if not new_robot_tasks and not new_robot2_tasks:
                    active[2] = False
                    print("🦾 Robot2 finished all task!")
                    break
            

            if not active[i]:
                continue
            
            # Set agents info.(task data & name)
            rep = c.replicants[i]
            if i == 0:
                task_list = robot_tasks
                task_path = "experiment/optimize_1/robot_task_list.json"
                agent_name = "🤖 Robot1"
            elif i == 2:
                task_list = robot2_tasks
                task_path = "experiment/optimize_1/robot2_task_list.json"
                agent_name = "🦾 Robot2"
            else:
                task_list = human_tasks
                task_path = "experiment/optimize_1/human_task_list.json"
                agent_name = "👋 Human"

            # 如果目前正在導航但發生碰撞，且尚未抵達目標位置，強制重試 navigate_to
            if rep.action and rep.action.status == ActionStatus.collision:
                last_index = indices[i] - 1
                if last_index >= 0:
                    last_action, last_target = task_list[last_index]
                    if last_action == "navigate_to" and isinstance(last_target, dict):
                        current_pos = {
                            "x": rep.dynamic.transform.position[0],
                            "y": rep.dynamic.transform.position[1],
                            "z": rep.dynamic.transform.position[2]
                        }
                        distance = np.linalg.norm(
                            np.array([current_pos["x"], current_pos["y"], current_pos["z"]]) -
                            np.array([last_target["x"], last_target["y"], last_target["z"]])
                        )
                        if distance > 0.15:
                            # print(f"⚠️ {agent_name} collided before reaching target. Retrying navigate_to...")
                            rep.action = None
                            indices[i] -= 1  # 回到該 navigate_to 任務
                            frame_count -= 1
                            if i == 0:
                                r0_frame_count -= 1
                            elif i == 2:
                                r2_frame_count -= 1
                            continue           

            if rep.action is None or rep.action.status != ActionStatus.ongoing:
                # 嘗試執行下一任務
                while indices[i] < len(task_list):
                    action, target = task_list[indices[i]]
                    # prev_action = task_list[indices[i] - 1]
                    prev_action = task_list[indices[i] - 1] if indices[i] > 0 else None
                    transported = Arm.right in rep.dynamic.held_objects
                    # print(action, prev_action, Arm.right in rep.dynamic.held_objects, transported)
                    
                    # 如果此 task 對應 food id 已被別人完成，直接跳過
                    # if action == "navigate_to" and indices[i] % 4 == 0:
                    #     # get food id
                    #     next_action = task_list[indices[i] + 1]
                    #     food_id = next_action[1]
                    #     # 檢查是否已被其他 agent 完成
                    #     with open(task_path, "r") as f:
                    #         tasks = json.load(f)
                    #     finished_ids = [t["id"] for t in tasks if t["state"] == "finished"]

                    #     if food_id in finished_ids:
                    #         print(f"❗ The food with ID {food_id} has already been completed by another agent during navigate_to.")
                    #         print(f"{agent_name} jump to next task!!!")
                    #         indices[i] += 4  # 跳過 navigate_to + drop
                    #         continue
                    
                    # [navigate_to、 "pick_up"、"navigate_to" 、drop]
                    # 判斷 food 是否被其他 agent 完成了 (human vs robot)
                    

                    if action == "navigate_to" and prev_action is not None and isinstance(prev_action, tuple) \
   and prev_action[0] == "pick_up" and transported == False:
                        if i in [0, 2]:  # robot1 or robot2
                            print(f"{agent_name} finds the food with ID {prev_action[1]} has already been completed by 👋 Human !!!")
                            mark_task_finished_both(prev_action[1])
                            # CALL LLM replan robots task(tasks finished by human)
                            print("Replanning task list!!!")
                            if i == 0 or i == 2:
                                human_interference = True
                                indices[i] = len(task_list)
                                continue
                                # 跳到最上面 human interference
                            indices[i] += 2
                        else:
                            print(f"The food with ID {prev_action[1]} has already been completed by other 🤖 Robots !!!")
                            mark_task_finished(prev_action[1], task_path)
                            indices[i] += 2
                        continue

                    executed = issue_action(rep, action, i, target)
                    indices[i] += 1
                    if executed:
                        break
                    else:
                        indices[i] += 2
                        continue
                else:
                    if not human_interference:
                        active[i] = False
                        if i == 0:
                            print("🤖 Robot1 finished all task!")
                        elif i == 1:
                            print("👋 Human finished all task!")
                        elif i == 2:
                            print("🦾 Robot2 finished all task!")

    print_scene_object_container(c)
    print_finished_foods()
    print(f"✅ Total frames used: {frame_count}")
    print(f"🤖 Robot1 used: {r0_frame_count}")
    print(f"🦾 Robot2 used: {r2_frame_count}")
    print("======================================================================")
    reset_robot_task_list("experiment/optimize_1/robot_task_list.json")
    reset_robot_task_list("experiment/optimize_1/robot2_task_list.json")


    # 記錄 total frames
    with open("experiment/optimize_1/frame_log-random.txt", "a") as log:
        log.write(f"{frame_count}\n")

    c.communicate({"$type": "terminate"})


if __name__ == "__main__":
    # 將 task_list.json 覆蓋到 robot_task_list.json
    shutil.copyfile('experiment/optimize_1/task_list.json', 'experiment/optimize_1/robot_task_list.json')
    shutil.copyfile('experiment/optimize_1/task_list.json', 'experiment/optimize_1/robot2_task_list.json')
    senario()

