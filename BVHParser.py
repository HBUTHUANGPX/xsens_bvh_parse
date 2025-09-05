
import re
class Node:
    def __init__(self, name, offset=None, channels=None, is_end=False):
        self.name = name
        self.offset = offset if offset is not None else [0.0, 0.0, 0.0]
        self.channels = channels if channels is not None else []
        self.children = []
        self.is_end = is_end

    def __str__(self, level=0):
        ret = (
            "  " * level
            + f"Node: {self.name}, Offset: {self.offset}, Channels: {self.channels}, Is_End: {self.is_end}\n"
        )
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


class BVHParser:
    def __init__(self):
        self.root = None
        self.frames = []
        self.frame_time = 0.0
        self.num_frames = 0
        self.channel_map = []  # 存储每个节点的通道索引

    def parse(self, text):
        lines = text.strip().split("\n")
        stack = []
        i = 0
        mode = "HIERARCHY"  # 切换 HIERARCHY 和 MOTION 模式

        # 解析 HIERARCHY
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith("MOTION"):
                mode = "MOTION"
                i += 1
                break
            try:
                if line.startswith("ROOT"):
                    name = line.split()[1]
                    self.root = Node(name)
                    stack.append(self.root)
                    self.channel_map.append((self.root, 0))
                elif line.startswith("JOINT"):
                    name = line.split()[1]
                    if not stack:
                        raise ValueError(
                            f"JOINT {name} found before ROOT or outside hierarchy"
                        )
                    node = Node(name)
                    stack[-1].children.append(node)
                    stack.append(node)
                    self.channel_map.append((node, 0))
                elif line.startswith("End Site"):
                    if not stack:
                        raise ValueError(
                            "End Site found before ROOT or outside hierarchy"
                        )
                    node = Node("EndSite", is_end=True)
                    stack[-1].children.append(node)
                    stack.append(node)
                elif line.startswith("OFFSET"):
                    parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.\d*", line)
                    if len(parts) != 3:
                        raise ValueError(f"Invalid OFFSET format in line: {line}")
                    offset = [float(p) for p in parts]
                    if not stack:
                        raise ValueError("OFFSET found before any node")
                    stack[-1].offset = offset
                elif line.startswith("CHANNELS"):
                    parts = line.split()
                    num = int(parts[1])
                    channels = parts[2 : 2 + num]
                    if len(channels) != num:
                        raise ValueError(f"CHANNELS count mismatch in line: {line}")
                    if not stack:
                        raise ValueError("CHANNELS found before any node")
                    stack[-1].channels = channels
                    # 更新 channel_map 中的通道起始索引
                    if stack[-1] is not self.root and not stack[-1].is_end:
                        self.channel_map[-1] = (
                            stack[-1],
                            self.channel_map[-1][1] + len(channels),
                        )
                elif line == "{":
                    pass
                elif line == "}":
                    if not stack:
                        raise ValueError("Unmatched closing brace '}'")
                    stack.pop()
                elif line.startswith("HIERARCHY"):
                    pass
                else:
                    raise ValueError(f"Unrecognized line in HIERARCHY: {line}")
            except Exception as e:
                raise ValueError(f"Error parsing HIERARCHY line {i+1}: {line}\n{e}")
            i += 1

        if stack:
            raise ValueError(
                "HIERARCHY parsing incomplete: stack not empty, missing closing braces"
            )
        if not self.root:
            raise ValueError("No ROOT node found in hierarchy")

        # 解析 MOTION
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                if line.startswith("Frames:"):
                    self.num_frames = int(line.split()[1])
                elif line.startswith("Frame Time:"):
                    self.frame_time = float(line.split()[2])
                else:
                    # 解析帧数据
                    parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.\d*", line)
                    frame_data = [float(p) for p in parts]
                    self.frames.append(frame_data)
            except Exception as e:
                raise ValueError(f"Error parsing MOTION line {i+1}: {line}\n{e}")
            i += 1

        if len(self.frames) != self.num_frames:
            raise ValueError(
                f"MOTION data has {len(self.frames)} frames, but expected {self.num_frames}"
            )

        return self.root, self.frames, self.frame_time

    def generate_mujoco_xml(self, scale=0.01, frame_0=[]):
        self.end_site = 0

        def generate_xml(node, indent=2):
            spaces = " " * indent
            if node.name == "Hips":
                pos = [frame_0[2], frame_0[0], frame_0[1] + 20]  # Z, X, Y
                print(pos)
                pos_str = " ".join(f"{x*scale:.6f}" for x in pos)
            else:
                pos = [node.offset[2], node.offset[0], node.offset[1]]  # Z, X, Y
                pos_str = " ".join(f"{x*scale:.6f}" for x in pos)

            xml = f'{spaces}<body name="{node.name}" pos="{pos_str}">\n'
            if node.name == "Hips":  # Root
                xml += f'{spaces}  <joint name="floating_base_joint" type="free" limited="false" actuatorfrclimited="false"/>\n'
            else:
                xml += f'{spaces}  <joint type="ball" name="{node.name}_joint"/>\n'
            xml += (
                f'{spaces}  <geom type="sphere" size="0.05" rgba="0.8 0.8 0.8 0.5"/>\n'
            )
            for child in node.children:
                if child.is_end:
                    pos = [node.offset[2], node.offset[0], node.offset[1]]  # Z, X, Y
                    end_pos = " ".join(f"{x*scale:.6f}" for x in pos)
                    xml += f'{spaces}  <site name="{child.name+str(self.end_site)}" pos="{end_pos}"/>\n'
                    self.end_site += 1
                else:
                    xml += generate_xml(child, indent + 2)
            xml += f"{spaces}</body>\n"
            return xml

        xml_header = """<mujoco model="human_skeleton">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
"""
        xml_footer = """  </worldbody>
        """
        xml_end = """
</mujoco>
"""
        body_xml = generate_xml(self.root, 4)
        scene = """
    <!-- setup scene -->
  !-- setup scene -->
  <statistic center="1.0 0.7 1.0" extent="0.8"/>
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="-140" elevation="-20" offwidth="2080" offheight="1170"/>
    </visual>
    <asset>
         <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture name="texplane" builtin="checker" height="512" width="512" rgb1=".2 .3 .4" rgb2=".1 .15 .2" type="2d" />
        <material name="MatPlane" reflectance="0.5" shininess="0.01" specular="0.1" texrepeat="1 1" texture="texplane" texuniform="true" />
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1"
      friction="0.6" condim="3"/>
      <!-- <light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1" castshadow="true"/> -->

          <light diffuse=".5 .5 .5" pos="-3 -3 5" dir="3 3 -5" castshadow="true"/>


    </worldbody>
"""
        return xml_header + body_xml + xml_footer + scene + xml_end
