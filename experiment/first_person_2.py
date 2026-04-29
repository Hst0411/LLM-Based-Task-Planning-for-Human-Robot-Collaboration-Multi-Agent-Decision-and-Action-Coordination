from tdw.replicant.action_status import ActionStatus
from tdw.add_ons.image_capture import ImageCapture
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.arm import Arm
import subprocess
from experiment.LLM_task_planner import get_robot_tasks_from_llm
import json
import numpy as np
import random
import os, glob

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

def sync_camera_to_replicant(c, replicant, height_offset=1.88, forward_distance=0.3, fov=90):
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

def mark_task_finished(food_id: int, task_list_path="experiment/robot_task_list.json"):
    with open(task_list_path, "r") as f:
        task_list = json.load(f)
    updated = False
    for item in task_list:
        if item["id"] == food_id:
            item["state"] = "finished"
            updated = True
            break
    if updated:
        with open(task_list_path, "w") as f:
            json.dump(task_list, f, separators=(',', ':'))
        print(f"✅ Marked {food_id} as finished in {task_list_path}")
    else:
        print(f"🚫 Warning: food_id {food_id} not found in {task_list_path}")

def is_close(pos1, pos2, threshold=0.03):
    a = np.array([pos1["x"], pos1["y"], pos1["z"]])
    b = np.array([pos2["x"], pos2["y"], pos2["z"]])
    return np.linalg.norm(a - b) < threshold

# reset task state
def reset_robot_task_list(task_list_path="experiment/robot_task_list.json"):
    # 讀取 JSON 檔案
    with open(task_list_path, 'r') as file:
        task_list = json.load(file)

    # 更新 task_list 中所有項目的 state 為 "unfinished"
    for task in task_list:
        task["state"] = "unfinished"

    # 儲存更新後的 task_list 回檔案
    with open(task_list_path, 'w') as file:
        json.dump(task_list, file, indent=4)

# ===== 模組化任務：導航、撿起、放下 =====
def issue_action(replicant, action_name, index, target=None):
    is_robot = index == 0
    if is_robot:
        agent_name = "🤖 Robot1"
        task_path = "experiment/robot_task_list.json"
    else:
        agent_name = "👋 Human"
        task_path = "experiment/human_task_list.json"

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
            #     print(f"🚫 Skipped navigation: already close to target {target}")

            #     # 嘗試標記該目標為 finished（回推該任務的 food_id）
            #     with open(task_path, "r") as f:
            #         task_list = json.load(f)
            #     for task in task_list:
            #         if is_close(task["target"], target):
            #             mark_task_finished(task["id"], task_path)
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
                mark_task_finished(food_id, task_path)

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
def get_human_tasks(task_list_path="experiment/human_task_list.json"):
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

def generate_human_task_list(all_tasks_path="experiment/task_list.json",
                              output_path="experiment/human_task_list.json",
                              num_items=1):
    with open(all_tasks_path, 'r') as f:
        all_tasks = json.load(f)
    selected = random.sample(all_tasks, num_items)
    with open(output_path, 'w') as f:
        json.dump(selected, f, indent=4)
    
    print("======================================================================")
    print(f"✅ Human randomly selected {num_items} tasks.")

# 顯示 robot_task_list.json 中完成的 food ID
def print_finished_foods(task_list_path="experiment/robot_task_list.json"):
    with open(task_list_path, "r") as f:
        task_list = json.load(f)
    
    finished = [item["id"] for item in task_list if item["state"] == "finished"]

    print(f"✅ Finished food items in {task_list_path}:")
    for i, food_id in enumerate(finished):
        print(f"{i+1}. {food_id}")
    print(f"🎯 Total delivered: {len(finished)} / {len(task_list)}")
    print("======================================================================")

def senario():
    c = TransportChallenge(screen_width=1280, screen_height=720)
    replicant_position = {"x": -3.8, "y": 0, "z": 3.46}
    human_position = {"x": -9.0, "y": 0, "z": 2.3}
    c.start_floorplan_trial(scene="2a", layout=0, replicants=[replicant_position, human_position],
                            num_containers=0, num_target_objects=8, random_seed=1)
    
    image_folder = "experiment/robot_first_person/human_robot_2/a"
    # 找出資料夾內所有 .jpg 檔案
    jpg_files = glob.glob(os.path.join(image_folder, "*.jpg"))

    # 刪除每一個 .jpg 檔案
    for file_path in jpg_files:
        os.remove(file_path)

    # 加入 image capture 模組，並設定 camera 為 avatar "a"
    capture = ImageCapture(path="experiment/robot_first_person/human_robot_2", avatar_ids=["a"])
    c.add_ons.append(capture)

    c.communicate([{"$type": "set_floorplan_roof", "show": False}])

    # 儲存路徑
    print("------------------------------------------------")
    print(f"Images will be saved to: experiment/robot_first_person/human_robot_2/a")
    print("------------------------------------------------")


    # print scene object
    print_scene_object_container(c)

    # robot Disable collision detection
    c.replicants[0].collision_detection.avoid = False
    # human Disable collision detection
    c.replicants[1].collision_detection.avoid = False

    print("GPT-4 開始規劃初始運輸順序")
    # robot_tasks 由 LLM 產生(T012489)
    robot_tasks = get_robot_tasks_from_llm(robot_position=replicant_position)
    
    # # 隨機產生 human task
    # generate_human_task_list()

    # # human_tasks 手動設計
    # human_tasks = get_human_tasks()

    # human_tasks 手動設計
    human_tasks = []

    indices = [0, 0]
    active = [True, True]
    frame_count = 0

    while any(active):
        c.communicate([])
        if active[0]:
            frame_count += 1
        sync_camera_to_replicant(c, c.replicants[0])
        
        for i in [0, 1]:
            if not active[i]:
                continue
            
            # Set agents info.(task data & name)
            rep = c.replicants[i]
            if i == 0:
                task_list = robot_tasks
                task_path = "experiment/robot_task_list.json"
                agent_name = "🤖 Robot1"
            else:
                task_list = human_tasks
                task_path = "experiment/human_task_list.json"
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
                    # print(action, prev_action, Arm.right in rep.dynamic.held_objects, transported)
                    
                    # 如果此 task 對應 food id 已被別人完成，直接跳過
                    if action == "navigate_to" and indices[i] % 4 == 0:
                        # get food id
                        next_action = task_list[indices[i] + 1]
                        food_id = next_action[1]
                        # 檢查是否已被其他 agent 完成
                        with open(task_path, "r") as f:
                            tasks = json.load(f)
                        finished_ids = [t["id"] for t in tasks if t["state"] == "finished"]

                        if food_id in finished_ids:
                            print(f"❗ The food with ID {food_id} has already been completed by another agent during navigate_to.")
                            print(f"{agent_name} jump to next task!!!")
                            indices[i] += 4  # 跳過 navigate_to + drop
                            continue
                    
                    # [navigate_to、 "pick_up"、"navigate_to" 、drop]
                    # 判斷 food 是否被其他 agent 完成了 (human vs robot)
                    if action == "navigate_to" and prev_action is not None and isinstance(prev_action, tuple) \
   and prev_action[0] == "pick_up" and transported == False:
                        if i in [0, 2]:  # robot1 or robot2
                            print(f"{agent_name} finds the food with ID {prev_action[1]} has already been completed by 👋 Human !!!")
                            mark_task_finished(prev_action[1], task_path)
                            # CALL LLM replan robots task(tasks finished by human)
                            print("Replanning task list!!!")
                            # replicant_position = c.replicants[0].dynamic.transform.position
                            # replicant2_position = c.replicants[2].dynamic.transform.position
                            # replicant_position, replicant2_position = transform_to_dict(replicant_position, 
                            #                                                             replicant2_position)
                            # new_robot_tasks, new_robot2_tasks, previous_plan = get_human_finished_tasks(replicant_position, 
                            #                                                                             replicant2_position, 
                            #                                                                             history=previous_plan)
                            # if i == 0:
                            #     # robot1 直接用新工作覆蓋，並從頭執行 (indices[0] = 0)
                            #     indices[i] = 0
                            #     task_list = new_robot_tasks
                                
                            #     # robot2 拿了東西 (第二個 navigate_to)，new_robot2_tasks 加在下個工作
                            #     robot2_hold_object = Arm.right in c.replicants[2].dynamic.held_objects
                            #     if robot2_hold_object:
                            #         robot2_tasks = robot2_tasks[:indices[2] + 2] + new_robot2_tasks
                            #     # 沒拿就直接把 new_robot2_tasks 全部覆蓋
                            #     else:
                            #         robot2_tasks = new_robot2_tasks
                            #         indices[2] = 0

                            # elif i == 2:
                            #     # robot2 直接用新工作覆蓋，並從頭執行 (indices[2] = 0)
                            #     indices[i] = 0
                            #     task_list = new_robot2_tasks
                                
                            #     # robot0 拿了東西 (第二個 navigate_to)，new_robot0_tasks 加在下個工作
                            #     robot_hold_object = Arm.right in c.replicants[0].dynamic.held_objects
                            #     if robot_hold_object:
                            #         robot_tasks = robot_tasks[:indices[0] + 2] + new_robot_tasks
                            #     # 沒拿就直接把 new_robot_tasks 全部覆蓋
                            #     else:
                            #         robot_tasks = new_robot_tasks
                            #         indices[0] = 0
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
                    active[i] = False
                    if i == 0:
                        print("🤖 Robot1 finished all task!")
                    elif i == 1:
                        print("👋 Human finished all task!")

    print_scene_object_container(c)
    print_finished_foods()
    print(f"✅ Total frames used: {frame_count}")
    print("======================================================================")
    reset_robot_task_list()

    # 記錄 total frames
    with open("experiment/totalframes_5_2-1/frame_log.txt", "a") as log:
        log.write(f"{frame_count}\n")

    c.communicate({"$type": "terminate"})


if __name__ == "__main__":
    senario()
