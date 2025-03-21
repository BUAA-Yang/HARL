import numpy as np

def angle_normalize(angle):
    """规范化角度到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_relative_bearing_and_distance(agent_pos, target_pos, omega):
    """
    计算目标相对智能体的水平方位角、水平距离和垂直距离

    参数:
    - agent_pos: 智能体的位置 [x, y, z]
    - target_pos: 目标的位置 [x, y, z]
    - omega: 智能体的水平朝向角 (弧度)

    返回:
    - relative_bearing: 目标相对于智能体的水平方位角 [-pi, pi] (弧度)
    - horizontal_distance: xy 平面上的水平距离
    - vertical_distance: z 方向的垂直距离
    """
    # 计算相对位置向量
    delta_pos = np.array(target_pos) - np.array(agent_pos)
    # 水平距离（忽略 z）
    horizontal_distance = np.linalg.norm(delta_pos[:2])  # sqrt(dx^2 + dy^2)
    # 垂直距离（z轴方向）
    vertical_distance = delta_pos[2]
    # 计算绝对水平方向（xy平面上的方位角）[-pi, pi]
    absolute_bearing = np.arctan2(delta_pos[1], delta_pos[0])
    # 计算相对方位角（目标相对 agent 当前朝向 omega 的水平角度） ∈ [-pi, pi]
    relative_bearing = absolute_bearing - omega
    # 规范化到 [-pi, pi]，保持一致性
    relative_bearing = (relative_bearing + np.pi) % (2 * np.pi) - np.pi
    return relative_bearing, horizontal_distance, vertical_distance
agent_pos = [0.0, 0.0, 10.0]   # 智能体位置 (x, y, z)
target_pos = [10.0, 10.0, 5.0] # 目标位置 (x, y, z)
omega = np.pi*(1+2/3)         # 智能体朝向 45 度，转换为弧度

bearing, h_dist, v_dist = compute_relative_bearing_and_distance(agent_pos, target_pos, omega)
print(f"相对方位角 (弧度): {bearing:.3f}")
print(f"相对方位角 (角度): {np.degrees(bearing):.2f}°")
print(f"水平距离: {h_dist:.3f}")
print(f"垂直距离: {v_dist:.3f}")
