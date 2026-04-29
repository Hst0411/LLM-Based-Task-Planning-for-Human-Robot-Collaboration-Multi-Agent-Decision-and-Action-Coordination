# 2 agents(2r)
# 計算兩 robots 花費的 frames
# 2 robots 已裝上 camera
# 協作策略：human 6 tasks 隨機排順序 、robots 每完成一個工作 or 發現 human 打亂 ---> LLM
# bug:
# 1. 第一筆沒拿到踢走的話之後會找不到(發生機率很低)
# 2. 剩最後一筆 task 原本分給 r1，r1 也正在前往，但 r2 重新分配(human_inference)後把這筆 task 分給自己(比較近)，r1 不確定是否會停下來
# 3. 一開始分配好的工作，r1 r2 拿起第一件工作後，任一 robot 一收到重新分配的工作，另一 robot 若當下在 drop 可能會因為 drop 還沒完成導致直接前往下個 task。
from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.image_capture import ImageCapture
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.arm import Arm
from experiment.optimize_1.LLM_task_planner import get_two_robot_tasks_from_llm
from experiment.optimize_1.human_finished_replanner import get_human_finished_tasks
from experiment.optimize_1.LLM_replan_anytime import report_progress_to_LLM
from experiment.robot_pereception_module import detect_and_check
from experiment.robot2_preception_module import detect_and_check2
import json
import numpy as np
import random
from typing import Union
import shutil, os, glob, sys

# 從 command-line 取得模型名稱，預設 gpt-4
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt-4"

def transform_to_dict(replicant_position, replicant2_position):
    replicant_position_dict = {
        'x': float(replicant_position[0]),
        'y': float(replicant_position[1]),
        'z': float(replicant_position[2])
    }
    replicant2_position_dict = {
        'x': float(replicant2_position[0]),
        'y': float(replicant2_position[1]),
        'z': float(replicant2_position[2])
    }
    return replicant_position_dict, replicant2_position_dict

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

# def sync_camera_to_replicant(c, replicant, height_offset=1.0, forward_distance=0.3, fov=90):
#     """
#     讓 camera avatar 移動到 replicant 頭部，並朝向他面對的方向，並調整 FOV。
#     """
#     rep_pos = replicant.dynamic.transform.position
#     forward = replicant.dynamic.transform.forward

#     cam_pos = {
#         "x": float(rep_pos[0]),
#         "y": float(rep_pos[1]) + height_offset,
#         "z": float(rep_pos[2]) - 0.5
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
#             "avatar_id": "a"
#         },
#         {
#             "$type": "look_at_position",
#             "position": look_at,
#             "avatar_id": "a"
#         },
#         {
#             "$type": "set_field_of_view",
#             "field_of_view": fov,   # 加上這行：設定攝影機視野大小
#             "avatar_id": "a"
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

def mark_task_finished(food_id: Union[int, str], robot_index: int, task_list_path="experiment/optimize_1/task_list_new.json"):
    robot_id = "robot1"
    agent_name = "🤖 Robot1"
    if robot_index == 2:
        robot_id = "robot2"
        agent_name = "🦾 Robot2"

    with open(task_list_path, "r") as f:
        task_list = json.load(f)

    task_target_pos = None

    # 遍歷 tasks 找到對應的任務
    for task in task_list["tasks"]:
        if (isinstance(food_id, int) and task["id"] == food_id) or \
           (isinstance(food_id, str) and task["name"] == food_id):
            task["state"] = "finished"
            task_target_pos = task["target"]
            break

    # 更新 robot 位置
    if robot_id in task_list["robots"]:
        task_list["robots"][robot_id]["position"] = task_target_pos
        # task_list["robots"][robot_id]["status"] = "idle"
        # task_list["robots"][robot_id]["current_task_id"] = None

    # 儲存更新後的 json 檔案
    with open(task_list_path, "w") as f:
        json.dump(task_list, f, indent=2)

    print(f"✅ {food_id} picked up by {agent_name}. Mark task '{food_id}' as finished and updated {robot_id}'s position.")

def mark_human_task_finished(food_name: Union[int, str], robot_index: int, task_list_path="experiment/optimize_1/task_list_new.json"):
    agent_name = "🤖 Robot1"
    if robot_index == 2:
        agent_name = "🦾 Robot2"

    with open(task_list_path, "r") as f:
        task_list = json.load(f)

    food_id = 0

    # 遍歷 tasks 找到對應的任務
    for task in task_list["tasks"]:
        if (isinstance(food_name, str) and task["name"] == food_name):
            food_id = task["id"]
            task["state"] = "finished"
            break

    # 儲存更新後的 json 檔案
    with open(task_list_path, "w") as f:
        json.dump(task_list, f, indent=2)
    
    print(f"✅ {agent_name} sees that the human is transporting {food_name} (ID = {food_id}). Mark task '{food_id}' as finished")

# ===== 模組化任務：navigate_to、pick_up、navigate_to、drop =====
def issue_action(replicant, action_name, index, target=None):
    # set agents task list path
    is_robot = index == 0
    is_robot2 = index == 2
    if is_robot:
        agent_name = "🤖 Robot1"
    elif is_robot2:
        agent_name = "🦾 Robot2"
    else:
        agent_name = "👋 Human"

    if action_name == "navigate_to":
        # if target is None:
        #     print("\u26a0\ufe0f Warning: navigate_to called with None target.")
        #     return False
        # if isinstance(target, dict) and all(k in target for k in ("x", "y", "z")):
        #     replicant.navigate_to(target)
        # else:
        #     print(f"\u26a0\ufe0f Warning: Invalid navigation target: {target}")
        #     return False
        replicant.navigate_to(target)

    elif action_name == "pick_up":

        if index in [0, 2]:  # robot1 or robot2
            mark_task_finished(food_id=target, robot_index=index)
        else:
            print(f"✅ {target} picked up by {agent_name}.")

        replicant.pick_up(target)
    elif action_name == "drop":
        replicant.drop(arm=Arm.right)

    return True


# ---------------------
# robot task lists(LLM)
# ---------------------
def LLM_robot_tasks(c):
    return [
        ("navigate_to", c.state.target_object_ids[0]),
        ("pick_up", c.state.target_object_ids[0])
        # ("go_to", c.state.target_object_ids[9]),
        # ("pick_up", c.state.target_object_ids[9])
        #("go_to", {"x": -9.23, "y": 0.086, "z": -0.69}),
        #("drop", None)
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

def generate_human_task_list(all_tasks_path="experiment/optimize_1/task_list_all.json",
                              output_path="experiment/optimize_1/human_task_list.json"): 
    # num_items = random.randint(1, 6)  # ✅ 隨機選 1~6 個
    num_items = 1

    with open(all_tasks_path, 'r') as f:
        all_tasks = json.load(f)

    selected = random.sample(all_tasks, num_items)

    with open(output_path, 'w') as f:
        json.dump(selected, f, indent=4)
    
    print("======================================================================")
    print(f"✅ Human randomly selected {num_items} tasks.")

# 顯示 robot_task_list.json 中完成的 food ID
def print_finished_foods(task_list_path="experiment/optimize_1/task_list_new.json"):
    with open(task_list_path, "r") as f:
        task_list = json.load(f)
    
    finished = [item["id"] for item in task_list["tasks"] if item["state"] == "finished"]

    print(f"✅ Finished food items in {task_list_path}:")
    for i, food_id in enumerate(finished):
        print(f"{i+1}. {food_id}")
    print(f"🎯 Total delivered: {len(finished)} / {len(task_list['tasks'])}")
    print("======================================================================")

def scenario(model_name="gpt-4"):
    global MODEL_NAME
    print(f"🎯 Running scenario with model: {model_name}")
    c = TransportChallenge(screen_width=1280, screen_height=720)
    replicant_position = {"x": -3.8, "y": 0, "z": 3.46}
    replicant2_position = {"x": 4.0, "y": 0, "z": 2.72}
    human_position = {"x": -8.5, "y": 0, "z": 2.0}
    c.start_floorplan_trial(scene="2a", layout="0", replicants=[replicant_position, human_position, replicant2_position],
                            num_containers=0, num_target_objects=8, random_seed=1)

    print_scene_object_container(c)
    # 等待 replicants spawn 完成
    resp = c.communicate([])
    print("✅ Scene and replicants spawned successfully.")
        
    # 🔥 必須明確建立兩個 avatar
    c.communicate([
        {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "r0"},
        {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "r2"}
    ])
    print("✅ Created two cameras (r0, r2)")

    # TDW 只能加入一個 image capture 模組，預設是顯示 avatar "a" 的相機畫面
    # 分別儲存兩 camera 拍到的畫面
    capture = ImageCapture(path="experiment/multiple_camera", avatar_ids=["r0", "r2"])
    c.add_ons.append(capture)
    print("✅ Added ImageCapture module")

    # 🔥 加這段，確保 replicants 全部 spawn 完
    for i in range(5):
        resp = c.communicate([])
        print(f"warmup frame {i+1} ok")
    
    print("✅ Warmup done.")

    # 找出資料夾內所有 .jpg 檔案
    image_folder1 = "experiment/multiple_camera/r0"
    image_folder2 = "experiment/multiple_camera/r2"

    # # 刪除每一個 .jpg 檔案
    for f in glob.glob(os.path.join(image_folder1, "*.jpg")):
        os.remove(f)
    for f in glob.glob(os.path.join(image_folder2, "*.jpg")):
        os.remove(f)
    
    print("✅ The folder has been completely cleaned up!!!")
    c.communicate([{"$type": "set_floorplan_roof", "show": False}])

    # 儲存路徑
    print("------------------------------------------------")
    print(f"Images will be saved to: experiment/multiple_camera/r0")
    print(f"Images will be saved to: experiment/multiple_camera/r2")
    print("------------------------------------------------")

    # print scene object
    # print_scene_object_container(c)

    # robot Disable collision detection
    c.replicants[0].collision_detection.avoid = False
    # human Disable collision detection
    c.replicants[1].collision_detection.avoid = False
    # robot Disable collision detection
    c.replicants[2].collision_detection.avoid = False

    # 初始同步 camera 位置
    # c.communicate([])  # 確保初始化
    # sync_camera_to_replicant(c, c.replicants[0])
    # sync_camera_to_replicant(c, c.replicants[2])
    
    print("GPT-4 開始規劃兩 robots 初始運輸順序")
    robot_tasks, robot2_tasks = get_two_robot_tasks_from_llm(
        task_list_path="experiment/optimize_1/task_list_new.json",
        robot1_position=replicant_position,
        robot2_position=replicant2_position,
        model_name=model_name
    )

    # 隨機產生 human task
    generate_human_task_list()

    # human_tasks 手動設計
    # human_tasks = get_human_tasks()
    human_tasks = []

    indices = [0, 0, 0]
    active = [True, True, True]
    r1_frame_count = 0  # robot1 frame cost
    r2_frame_count = 0  # robot2 frame cost
    frame_count = 0     # total frame cost
    human_interference = False
    r1_last_idx = 0     # robot1 camera index
    r2_last_idx = 0     # robot2 camera index

    complete_task = False
    complete2_task = False
    new_robot_tasks = []
    new_robot2_tasks = []
    xx = 0
    last1 = 0
    last2 = 0

    while any(active):
        # c.communicate([])

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
            r1_frame_count += 1
            if r1_frame_count % 100 == 0:
                r1_last_idx, events = detect_and_check(start_index=r1_last_idx)
                for e in events:
                    print("🔔 Event:", e)

                    # 偵測到 human 運送 food(mark_task_list(food))，robot 跳下一個工作
                    if "detected a human carrying" in e:
                        food_name = e.split("detected a human carrying")[-1].strip()
                        print(f"🍎 Robot1 detected human carrying food: {food_name}")
                        mark_human_task_finished(food_name=food_name, robot_index=0)
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
                        mark_human_task_finished(food_name=food_name, robot_index=2)
                        # indices[2] += 4
        
        for i in [0, 1, 2]:
            # ------------------------------------------------------------------------------------------------------
            # human interference
            if human_interference:
                human_interference = False

                new_robot_tasks, new_robot2_tasks = get_human_finished_tasks(task_list_path="experiment/optimize_1/task_list_new.json", model_name=model_name)
                
                # 同步另一台 robots
                # if i == 1 and active[2]:
                #     while True:
                #         c.communicate([])
                #         if c.replicants[2].action.status in [ActionStatus.success, ActionStatus.ongoing]:
                #             break
                #     print(f"🦾 Robot2 executes action successfully.")
                # elif i == 0 and active[0]:
                #     while True:
                #         c.communicate([])
                #         if c.replicants[0].action.status in [ActionStatus.success, ActionStatus.ongoing]:
                #             break
                #     print(f"🤖 Robot1 executes action successfully.")

                # robot1 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects
                # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
                
                # robot1 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                if robot_hold_object:
                    print("robot1 is holding objects")
                    if indices[0] % 4 == 2:
                        robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
                    elif indices[0] % 4 == 3:
                        robot_tasks = robot_tasks[:indices[0] + 1] + new_robot_tasks
                    # print(robot_tasks)
                # 沒拿就直接把 new_robot_tasks 全部覆蓋
                else:
                    print("robot1 isn't holding objects")
                    robot_tasks = new_robot_tasks
                    indices[0] = 0
                
                # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                if robot2_hold_object:
                    print("robot2 is holding objects")
                    if indices[2] % 4 == 2:
                        robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
                    elif indices[2] % 4 == 3:
                        robot2_tasks = robot2_tasks[:indices[2] + 1] + new_robot2_tasks
                    # print(robot2_tasks)
                # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                else:
                    print("robot2 isn't holding objects")
                    robot2_tasks = new_robot2_tasks
                    indices[2] = 0

                if new_robot_tasks == []:
                    # 💤 Robot1 沒有分配新任務 → 保持 idle 狀態
                    c.replicants[0].action = None
                    with open("experiment/optimize_1/task_list_new.json", "r") as f:
                        task_state = json.load(f)
                    task_state["robots"]["robot1"]["status"] = "idle"
                    task_state["robots"]["robot1"]["current_task_id"] = None
                    with open("experiment/optimize_1/task_list_new.json", "w") as f:
                        json.dump(task_state, f, indent=2)
                    print("🤖 Robot1 currently has no task. Entering idle state.")
                    # 不 break，讓他留在場上等待下一次 replan
                    continue

                if new_robot2_tasks == []:
                    # 💤 Robot2 沒有分配新任務 → 保持 idle 狀態
                    c.replicants[2].action = None
                    with open("experiment/optimize_1/task_list_new.json", "r") as f:
                        task_state = json.load(f)
                    task_state["robots"]["robot2"]["status"] = "idle"
                    task_state["robots"]["robot2"]["current_task_id"] = None
                    with open("experiment/optimize_1/task_list_new.json", "w") as f:
                        json.dump(task_state, f, indent=2)
                    print("🦾 Robot2 currently has no task. Entering idle state.")
                    # 不 break，讓他留在場上等待下一次 replan
                    continue
            
            # ------------------------------------------------------------------------------------------------------
            # 🔥 robot1 完成，robot2 還有工作
            # if indices[0] >= len(robot_tasks) and active[0]:
            #     print("🤖 Robot1 finished all tasks → re-planning with LLM")
            #     replicant_position = c.replicants[0].dynamic.transform.position
            #     replicant2_position = c.replicants[2].dynamic.transform.position
            #     replicant_position, replicant2_position = transform_to_dict(replicant_position, replicant2_position)
            #     new_robot_tasks, new_robot2_tasks = get_global_plan_from_llm(replicant_position, replicant2_position,
            #                                                                  history=0)                                
            #     # robot1 直接用新工作覆蓋，並從頭執行 (indices[0] = 0)
            #     if new_robot_tasks:
            #         robot_tasks = new_robot_tasks
            #         indices[0] = 0
                
            #     # if new_robot2_tasks:
            #     # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
            #     robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
            #     if robot2_hold_object:
            #         print("robot2 is holding objects")
            #         if indices[2] % 4 == 2:
            #             robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
            #         elif indices[2] % 4 == 3:
            #             robot2_tasks = robot2_tasks[:indices[2] + 1] + new_robot2_tasks
            #     # 沒拿就直接把 new_robot2_tasks 全部覆蓋
            #     else:
            #         print("robot2 isn't holding objects")
            #         robot2_tasks = new_robot2_tasks
            #         indices[2] = 0
                
            #     # robot2 繼續工作，關掉 robot1
            #     if not new_robot_tasks:
            #         active[0] = False
            #         # print(robot_tasks)
            #         # print(indices[0])
            #         # indices[0] = len(robot_tasks)
            #         print("🤖 Robot1 finished all task!")
            #         break
                
            #     # if not new_robot_tasks and not new_robot2_tasks:
            #     #     active[0] = False
            #     #     print("🤖 Robot1 finished all task!")
            #     #     break
            
            # ------------------------------------------------------------------------------------------------------
            # 🔥 robot2 完成，robot1 還有工作
            # if indices[2] >= len(robot2_tasks) and active[2]:
            #     print("🦾 Robot2 finished all tasks → re-planning with LLM")
            #     replicant_position = c.replicants[0].dynamic.transform.position
            #     replicant2_position = c.replicants[2].dynamic.transform.position
            #     replicant_position, replicant2_position = transform_to_dict(replicant_position, replicant2_position)
            #     new_robot_tasks, new_robot2_tasks = get_global_plan_from_llm(replicant_position, replicant2_position)
            #     # robot2 直接用新工作覆蓋，並從頭執行 (indices[2] = 0)
            #     if new_robot2_tasks:
            #         robot2_tasks = new_robot2_tasks
            #         indices[2] = 0
                
            #     # if new_robot_tasks:
            #     # robot1 拿了東西 (第二個 navigate_to)，new_robot_tasks 加在下個工作
            #     robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects
            #     if robot_hold_object:
            #         print("robot1 is holding objects")
            #         if indices[0] % 4 == 2:
            #             robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
            #         elif indices[0] % 4 == 3:
            #             robot_tasks = robot_tasks[:indices[0] + 1] + new_robot_tasks
            #     # 沒拿就直接把 new_robot2_tasks 全部覆蓋
            #     else:
            #         print("robot1 isn't holding objects")
            #         robot_tasks = new_robot_tasks
            #         indices[0] = 0

            #     # robot1 繼續工作，關掉 robot2
            #     if not new_robot2_tasks:
            #         active[2] = False
            #         # print(robot2_tasks)
            #         # print(indices[2])
            #         # indices[2] = len(robot2_tasks)
            #         print("🦾 Robot2 finished all task!")
            #         break
                
            #     # if not new_robot_tasks and not new_robot2_tasks:
            #     #     active[2] = False
            #     #     print("🦾 Robot2 finished all task!")
            #     #     break

            if complete_task:
                complete_task = False
                # 回報進度給 LLM
                result, new_robot_tasks, new_robot2_tasks = report_progress_to_LLM(task_list_path="experiment/optimize_1/task_list_new.json", model_name=model_name)
                # 完全沒工作
                if result == False and new_robot_tasks == [] and new_robot2_tasks == []:
                    if i == 0:
                        active[2] = False
                        print("Shut down robot2 !!!")
                    elif i == 1:
                        active[0] = False
                        print("Shut down robot1 !!!")
                # 思考後決定調整工作
                if result is not None and result["replan"] == True: 
                    # 🛑 強制停止所有正在執行的舊動作，避免繼續跑舊任務
                    for i in [0, 2]:
                        rep_stop = c.replicants[i]
                        if rep_stop.action is not None:
                            print(f"🛑 Forcing {['🤖 Robot1', '🦾 Robot2'][i//2]} to stop current action before replanning.")
                            rep_stop.action = None
                            c.communicate([])  # 強制同步一次狀態

                    # robot1 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                    robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects
                    # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                    robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
                    
                    # robot1 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                    if robot_hold_object:
                        print("robot1 is holding objects")
                        if indices[0] % 4 == 2:
                            robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
                        elif indices[0] % 4 == 3:
                            robot_tasks = robot_tasks[:indices[0] + 1] + new_robot_tasks
                        # print(robot_tasks)
                    # 沒拿就直接把 new_robot_tasks 全部覆蓋
                    else:
                        print("robot1 isn't holding objects")
                        robot_tasks = new_robot_tasks
                        indices[0] = 0
                    
                    # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                    if robot2_hold_object:
                        print("robot2 is holding objects")
                        if indices[2] % 4 == 2:
                            robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
                        elif indices[2] % 4 == 3:
                            robot2_tasks = robot2_tasks[:indices[2] + 1] + new_robot2_tasks
                        # print(robot2_tasks)
                    # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                    else:
                        print("robot2 isn't holding objects")
                        robot2_tasks = new_robot2_tasks
                        indices[2] = 0
                # 思考後決定不調整
                else:
                    print(f"i == {i}")
                    #  robot 繼續執行原下個工作
                    if i == 1:
                        indices[0] = last1 + 1
                    elif i == 0:
                        indices[2] = last2 + 1

            if not active[i]:
                continue
            
            # Set agents info.(task data & name)
            rep = c.replicants[i]
            if i == 0:
                task_list = robot_tasks
                agent_name = "🤖 Robot1"
                robot_id = "robot1"
                task_list_path = "experiment/optimize_1/task_list_new.json"
            elif i == 2:
                task_list = robot2_tasks
                agent_name = "🦾 Robot2"
                robot_id = "robot2"
                task_list_path = "experiment/optimize_1/task_list_new.json"
            else:
                task_list = human_tasks
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
                            continue           

            if rep.action is None or rep.action.status != ActionStatus.ongoing:
                # 嘗試執行下一任務
                while indices[i] < len(task_list):
                    action, target = task_list[indices[i]]
                    # prev_action = task_list[indices[i] - 1]
                    prev_action = task_list[indices[i] - 1] if indices[i] > 0 else None
                    transported = Arm.right in rep.dynamic.held_objects

                    if action == "navigate_to" and prev_action is not None and isinstance(prev_action, tuple) \
   and prev_action[0] == "pick_up" and transported == False:
                        if i in [0, 2]:  # robot1 or robot2
                            print(f"{agent_name} finds the food with ID {prev_action[1]} has already been completed by 👋 Human !!!")
                            mark_task_finished(food_id=prev_action[1], robot_index=i)

                            # CALL LLM replan robots task(tasks finished by human)
                            print("Replanning task list!!!")
                            if i == 0 or i == 2:
                                human_interference = True
                                
                                # 回報位置和狀態
                                with open(task_list_path, "r") as f:
                                            task_list_new = json.load(f)
                                replicant_position = c.replicants[0].dynamic.transform.position
                                replicant2_position = c.replicants[2].dynamic.transform.position
                                replicant_position, replicant2_position = transform_to_dict(replicant_position, 
                                                                                            replicant2_position)
                                if i == 0:
                                    task_list_new["robots"]["robot2"]["position"] = replicant2_position
                                else:
                                    task_list_new["robots"]["robot1"]["position"] = replicant_position
                                # 儲存更新後的 json 檔案
                                with open(task_list_path, "w") as f:
                                    json.dump(task_list_new, f, indent=2)

                                indices[i] = len(task_list)
                                continue
                                # 跳到最上面 human interference
                            indices[i] += 2
                        else:
                            print(f"The food with ID {prev_action[1]} has already been completed by other 🤖 Robots !!!")
                            indices[i] += 2
                        continue

                    executed = issue_action(rep, action, i, target)
                    
                    # robot report progress to LLM
                    if action == "drop" and i in [0, 2]:
                        print(f"{agent_name} executing drop...")
                        # ✅ 等待 drop 動作真正完成
                        while True:
                            c.communicate([])
                            if rep.action.status in [ActionStatus.success, ActionStatus.ongoing]:
                                break
                        print(f"{agent_name} finished dropping item successfully.")

                        with open(task_list_path, "r") as f:
                            task_list_new = json.load(f)
                        
                        # 已經是最後一個工作了
                        if indices[i] + 1 == len(task_list):
                            # 更新 robot 狀態 --> "idle"
                            task_list_new["robots"][robot_id]["status"] = "idle"
                            task_list_new["robots"][robot_id]["current_task_id"] = None
                        # 之後還有工作
                        else:
                            # 更新 robot 狀態 --> "executing" 和 "next task id"
                            next_task = task_list[indices[i] + 2]
                            task_list_new["robots"][robot_id]["status"] = "executing"
                            task_list_new["robots"][robot_id]["current_task_id"] = next_task[1]
                        
                        print(f"{agent_name} is reporting progress to LLM.")
                        
                        # 呼叫 LLM 前再重新回報另一台 robot 位置
                        replicant2_position = c.replicants[2].dynamic.transform.position
                        replicant_position = c.replicants[0].dynamic.transform.position
                        replicant_position, replicant2_position = transform_to_dict(replicant_position, replicant2_position)
                        if i == 0:
                            task_list_new["robots"]["robot2"]["position"] = replicant2_position
                        else:
                            task_list_new["robots"]["robot1"]["position"] = replicant_position

                        # 儲存更新後的 json 檔案
                        with open(task_list_path, "w") as f:
                            json.dump(task_list_new, f, indent=2)
                        
                        complete_task = True
                        print(f"{xx} complete_task is: {complete_task}")
                        # c.communicate([])
                        last1 = indices[0]
                        last2 = indices[2]
                        indices[i] += 1
                        continue

                        # # 需要重新安排
                        # if result["replan"] == True:
                            
                        #     # robot1 繼續工作，關掉 robot2
                        #     if not new_robot_tasks:
                        #         active[0] = False
                        #         print("🤖 Robot1 finished all task!")

                        #     # robot1 繼續工作，關掉 robot2
                        #     if not new_robot2_tasks:
                        #         active[2] = False
                        #         print("🦾 Robot2 finished all task!")

                        #     if i == 0:
                        #         # robot1 直接覆蓋
                        #         robot_tasks = new_robot_tasks
                        #         indices[0] = 0
                        #         # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                        #         robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
                        #         if robot2_hold_object:
                        #             print("robot2 is holding objects")
                        #             if indices[2] % 4 == 2:
                        #                 robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
                        #             elif indices[2] % 4 == 3:
                        #                 robot2_tasks = robot2_tasks[:indices[2] + 1] + new_robot2_tasks
                        #         # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                        #         else:
                        #             print("robot2 isn't holding objects")
                        #             robot2_tasks = new_robot2_tasks
                        #             indices[2] = 0
                        #     else:
                        #         # robot2 直接覆蓋
                        #         robot2_tasks = new_robot2_tasks
                        #         indices[2] = 0
                        #         # robot1 拿了東西 (第二個 navigate_to)，new_robot_tasks 加在下個工作
                        #         robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects                                
                        #         if robot_hold_object:
                        #             print("robot1 is holding objects")
                        #             if indices[0] % 4 == 2:
                        #                 robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
                        #             elif indices[0] % 4 == 3:
                        #                 robot_tasks = robot_tasks[:indices[0] + 1] + new_robot_tasks
                        #         # 沒拿就直接把 new_robot_tasks 全部覆蓋
                        #         else:
                        #             print("robot1 isn't holding objects")
                        #             robot_tasks = new_robot_tasks
                        #             indices[0] = 0
                        #     continue

                    indices[i] += 1
                    if executed:
                        break
                    else:
                        indices[i] += 2
                        continue
                else:
                    with open(task_list_path, "r") as f:
                        task_list_new = json.load(f)
                    # 要全部 task state == "finished" 才可關掉 robot
                    if all(task["state"] == "finished" for task in task_list_new["tasks"]):
                        with open(task_list_path, "r") as f:
                            task_list_new = json.load(f)
                        
                        # 更新 robot 狀態 --> "idle"
                        task_list_new["robots"][robot_id]["status"] = "idle"
                        task_list_new["robots"][robot_id]["current_task_id"] = None

                        print(f"{agent_name} is idle.")

                        # 儲存更新後的 json 檔案
                        with open(task_list_path, "w") as f:
                            json.dump(task_list_new, f, indent=2)

                        # 最後關掉 agent
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
    print("======================================================================")

    # ✅ 記錄 total frames（自動依 model 寫入對應資料夾）
    log_dir = f"experiment/optimize_1/{model_name}/frame_log.txt"
    with open(log_dir, "a") as log:
        log.write(f"{frame_count}\n")

    c.communicate({"$type": "terminate"})


if __name__ == "__main__":
    
    # 預設 model 名稱
    model_name = "gpt-4"

    # ✅ 若有命令列參數（由 testscript 傳入），就使用該 model 名稱
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    # ✅ 自動建立對應資料夾
    log_folder = f"experiment/optimize_1/{model_name}"
    os.makedirs(log_folder, exist_ok=True)

    # 將 task_list_all_new.json 覆蓋到 task_list_new.json
    shutil.copyfile('experiment/optimize_1/task_list_all_new.json', 'experiment/optimize_1/task_list_new.json')
    scenario(model_name=model_name)