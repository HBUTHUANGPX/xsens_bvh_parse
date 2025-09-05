
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
import time

from BVHParser import BVHParser


def euler_to_quat(euler):
    """Convert Euler angles (in degrees) to quaternion in YXZ order."""
    # Convert degrees to radians
    bvh_euler = np.array(euler)
    Yrotation = bvh_euler[0]
    Xrotation = bvh_euler[1]
    Zrotation = bvh_euler[2]
    mujoco_euler = [Zrotation, Xrotation, Yrotation]
    # mujoco_euler = [0,0,0]
    mujoco_euler_rad = np.deg2rad(mujoco_euler)
    # Create rotation object with YXZ extrinsic order
    rot = Rotation.from_euler("xyz", mujoco_euler_rad, degrees=False)
    # print(rot.as_quat())
    # rot = Rotation.from_euler(order.lower(), mujoco_euler_rad, degrees=False)
    # Convert to quaternion (scalar-last format: [x, y, z, w])
    quat = rot.as_quat(scalar_first=True)#[[3, 0, 1, 2]]
    return quat


def animate_bvh(scale, bvh_file, xml_file):
    # 解析 BVH 文件
    parser = BVHParser()
    with open(bvh_file, "r") as f:
        root, frames, frame_time = parser.parse(f.read())

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)

    # 获取关节 ID 和通道映射
    joint_names = [node[0].name for node in parser.channel_map if not node[0].is_end]
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        for name in joint_names
    ]
    # 动画播放
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        while viewer.is_running() and frame_idx < len(frames):
            frame_data = frames[frame_idx]
            channel_idx = 0

            # 设置根节点 (Hips) 的位置和旋转
            # 位置 (cm to m, 坐标转换: X -> Y, Y -> Z, Z -> X)
            bvh_pos = np.array(frame_data[0:3]) * scale  # 缩放到米
            mujoco_pos = [bvh_pos[2], bvh_pos[0], bvh_pos[1]]  # Z, X, Y
            data.qpos[0:3] = mujoco_pos
            data.qvel[:] = 0
            # 旋转 (YXZ 欧拉角转换为四元数)
            euler = frame_data[3:6]  # Yrotation, Xrotation, Zrotation
            data.qpos[3:7] = euler_to_quat(
                euler
            )  # *0+np.array([1,0,0,0],dtype=np.float32)
            
            # 设置其他关节的旋转
            channel_idx = 6
            for joint_id, (node, _) in zip(joint_ids, parser.channel_map):
                if joint_id >= 0 and not node.is_end:
                    euler = frame_data[channel_idx : channel_idx + 3]  # YXZ 顺序
                    quat = euler_to_quat(euler)
                    qpos_idx = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_idx : qpos_idx + 4] = (
                        quat  # *0+np.array([1,0,0,0],dtype=np.float32)
                    )
                    channel_idx += 3
            print(data.qpos[3:])
            time.sleep(frame_time)
            mujoco.mj_step(model, data)
            viewer.sync()
            frame_idx += 1
            if frame_idx% len(frames)==0:
                break
            # frame_idx = (frame_idx + 1) % len(frames)  # 循环播放


if __name__ == "__main__":
    bvh_file_name = "xsens_bvh/xsens_walk.bvh"
    xml_file_name = "human_skeleton.xml"
    scale = 0.01
    with open(bvh_file_name, "r") as f:
        bvh_text = f.read()

    parser = BVHParser()
    root, frames, frame_time = parser.parse(bvh_text)
    # print("Parsed BVH:")
    # print(root)
    # print(f"Number of frames: {len(frames)}")
    # print(f"Frame time: {frame_time}")
    # print(f"First frame data: {frames[0]}")

    # 生成 MuJoCo XML
    xml_content = parser.generate_mujoco_xml(scale=scale, frame_0=frames[0])
    # print(xml_content)
    with open(xml_file_name, "w") as f:
        f.write(xml_content)
    # print("MuJoCo XML generated: human_skeleton.xml")

    animate_bvh(scale, bvh_file_name, xml_file_name)
