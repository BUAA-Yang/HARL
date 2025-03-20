from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apply Seaborn style
sns.set(style="white")     # White background style

# 无人机位置和速度矢量
drone_position = np.array([0, 0, 0])  # 无人机改为原点
horizontal_velocity = np.array([4, -5, 0])
vertical_velocity = np.array([0, 0, 4])
r=2
# 统一控制线条粗细和箭头大小
LINE_WIDTH = 2  # 统一线条粗细
ARROW_SIZE = 1.0  # 统一箭头大小

# 创建3D图形
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
h=4
# 绘制与XY轴平行的平面
x = np.linspace(-8, 8, 17)  # 间隔1单位
y = np.linspace(-8, 8, 13)  # 间隔1单位
x, y = np.meshgrid(x, y)
z = np.full_like(x, drone_position[2]-r-h) 
# ax.plot_surface(x, y, z, alpha=0.1, color='gray', edgecolor='black', linewidth=0.2)  # 边线较细
# 绘制保护区的圆形投影
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# 绘制保护区的圆形投影
theta = np.linspace(0, 2 * np.pi, 100)
x_proj = r * np.cos(theta) + drone_position[0]
y_proj = r * np.sin(theta) + drone_position[1]
z_proj = np.full_like(x_proj, drone_position[2] - r - h)  # 投影平面的z值

# 创建多边形面
verts = [list(zip(x_proj, y_proj, z_proj))]

# 创建Poly3DCollection对象进行填充
polygon = Poly3DCollection(verts, color='red', alpha=0.5)


# 绘制圆形边界
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 显示文字“保护区”
ax.text(drone_position[0]-0.5, drone_position[1]-0.5, drone_position[2] - r - h, '保护区', color='black', fontsize=12, ha='center')

# 绘制通信范围的圆形填充区
r_communication = 3 * r  # 通信范围半径
x_proj_comm = r_communication * np.cos(theta) + drone_position[0]
y_proj_comm = r_communication * np.sin(theta) + drone_position[1]
z_proj_comm = np.full_like(x_proj, drone_position[2] - r - h)  # 投影平面的z值

# 创建多边形面
verts_comm = [list(zip(x_proj_comm, y_proj_comm, z_proj_comm))]

# 创建Poly3DCollection对象进行填充
polygon_comm = Poly3DCollection(verts_comm, color='yellow', alpha=0.1)
# 创建Poly3DCollection对象进行填充
polygon = Poly3DCollection(verts, color='red', alpha=0.5)
ax.add_collection3d(polygon_comm)
ax.add_collection3d(polygon)


# 显示文字“通信范围”
ax.text(drone_position[0]-1.5*r-0.5, drone_position[1]-r-0.5, drone_position[2] - 1.5*r - h, '通信范围', color='black', fontsize=12, ha='center')
# 绘制球形保护区
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = r * np.outer(np.cos(u), np.sin(v)) + drone_position[0]
y = r * np.outer(np.sin(u), np.sin(v)) + drone_position[1]
z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + drone_position[2]
ax.plot_surface(x, y, z, color='gray', alpha=0.8, edgecolor='k', linewidth=0.1, zorder=10)
arrow_length_ratio=0.1
# 绘制XYZ轴，使用不同的箭头样式
ax.quiver(0, 0, 0, 10, 0, 0, color='black', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, label='X-axis')
ax.quiver(0, 0, 0, 0, 10, 0, color='black', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, label='Y-axis')
ax.quiver(0, 0, 0, 0, 0, 10, color='black', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, label='Z-axis')

# 在箭头终点添加Latex格式文字
ax.text(10.5, 0, 0, r'$X$', fontsize=14, color='black')
ax.text(0, 10.5, 0, r'$Y$', fontsize=14, color='black')
ax.text(0, 0, 10.5, r'$Z$', fontsize=14, color='black')

# 绘制无人机位置
ax.scatter(drone_position[0], drone_position[1], drone_position[2], color='red', label='Drone')

# 绘制速度矢量（红色，平行XY）
ax.quiver(drone_position[0], drone_position[1], drone_position[2],
          horizontal_velocity[0], horizontal_velocity[1], horizontal_velocity[2],
          color='red', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, length=ARROW_SIZE)

# 计算箭头终点坐标
arrow_end = drone_position + horizontal_velocity

# 在箭头末端添加LaTeX文字 v_{i}(t)
ax.text(arrow_end[0], arrow_end[1], arrow_end[2], r'$v_{i}(t)$', fontsize=12, color='red')

# --------- X方向分量箭头 ---------
x_arrow = np.array([horizontal_velocity[0], 0, 0])  # 只在X方向
x_end = drone_position + x_arrow

ax.quiver(*drone_position,
          *x_arrow,
          color='green', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, length=ARROW_SIZE)

ax.text(x_end[0], x_end[1]+1, x_end[2], r'$\dot{x}_{i}(t)$', fontsize=12, color='green')

# --------- Y方向分量箭头 ---------
y_arrow = np.array([0, horizontal_velocity[1], 0])  # 只在Y方向
y_end = drone_position + y_arrow

ax.quiver(*drone_position,
          *y_arrow,
          color='purple', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, length=ARROW_SIZE)

ax.text(y_end[0], y_end[1]-0.5, y_end[2], r'$\dot{y}_{i}(t)$', fontsize=12, color='purple')

vertical_arrow_end = drone_position + vertical_velocity

ax.quiver(drone_position[0], drone_position[1], drone_position[2],
          vertical_velocity[0], vertical_velocity[1], vertical_velocity[2],
          color='red', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, length=ARROW_SIZE)

ax.text(vertical_arrow_end[0], vertical_arrow_end[1], vertical_arrow_end[2],
        r'$\dot{z}_{i}(t)$', fontsize=12, color='red')

# --------- 同方向加速度箭头 ---------
# 1.8 倍的垂直速度矢量
hor_acc_vector = 1.8 * vertical_velocity
hor_acc_end = drone_position + hor_acc_vector

ax.quiver(drone_position[0], drone_position[1], drone_position[2],
          hor_acc_vector[0], hor_acc_vector[1], hor_acc_vector[2],
          color='blue', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, length=ARROW_SIZE)

ax.text(hor_acc_end[0], hor_acc_end[1], hor_acc_end[2],
        r'$\eta_{i}^{\mathrm{Hor}}$', fontsize=12, color='blue')

# 计算水平速度的方向角度（XY平面）
velocity_angle = np.arctan2(horizontal_velocity[1], horizontal_velocity[0])
# 水平速度模长
speed_magnitude = np.linalg.norm(horizontal_velocity)
# 弧形箭头参数
arc_radius = speed_magnitude / 2  # 圆弧半径
arc_center = drone_position
arc_height = drone_position[2]  # Z固定不变

# 角度范围（顺时针旋转1/3 pi）
theta_start = velocity_angle
theta_end = velocity_angle - np.pi / 2
theta = np.linspace(theta_start, theta_end, 100)

# 计算弧线上的点
arc_x = arc_center[0] + arc_radius * np.cos(theta)
arc_y = arc_center[1] + arc_radius * np.sin(theta)
arc_z = np.full_like(arc_x, arc_height)

# 绘制圆弧
ax.plot(arc_x, arc_y, arc_z, color='blue', linewidth=LINE_WIDTH)

# 箭头的位置，放在弧线的终点
arrow_theta = theta_end
arrow_pos = np.array([
    arc_center[0] + arc_radius * np.cos(arrow_theta),
    arc_center[1] + arc_radius * np.sin(arrow_theta),
    arc_height
])

# 箭头朝向（切线方向，顺时针）
arrow_dir = np.array([
    np.sin(arrow_theta),
    -np.cos(arrow_theta),
    0
])

# 绘制箭头（弧线末端箭头）
arrow_len = 0.5
ax.quiver(*arrow_pos,
          *(arrow_len * arrow_dir),
          color='blue', arrow_length_ratio=2, linewidth=LINE_WIDTH, length=ARROW_SIZE)

# 添加标签 ω_i(t)
label_theta = theta_start - (theta_start - theta_end) / 2  # 中间位置
label_pos = arrow_pos

ax.text(label_pos[0]-0.5, label_pos[1]-0.5, label_pos[2], r'$\omega_{i}(t)$',
        fontsize=12, color='blue')

# 加速度大小为 2/3，方向相反
acceleration_magnitude = (2 / 3) * speed_magnitude
acceleration_direction = -horizontal_velocity / speed_magnitude  # 单位方向
acceleration_vector = acceleration_magnitude * acceleration_direction

# 加速度箭头终点
acc_end = drone_position + acceleration_vector

# 绘制加速度箭头
ax.quiver(*drone_position,
          *acceleration_vector,
          color='blue', arrow_length_ratio=arrow_length_ratio, linewidth=LINE_WIDTH, length=ARROW_SIZE)

# 添加 LaTeX 标签
ax.text(acc_end[0]-1, acc_end[1]+1, acc_end[2],
        r'$\eta_{i}^{\mathrm{Ver}}$', fontsize=12, color='blue')

# 1. 从 v_i(t) 终点到 x 分量箭头终点，垂线 (连线两点)
ax.plot([arrow_end[0], arrow_end[0]],
        [arrow_end[1], x_end[1]],
        [arrow_end[2], x_end[2]],
        color='green', linestyle='dashed', linewidth=LINE_WIDTH)

# 2. 从 v_i(t) 终点到 y 分量箭头终点，垂线 (连线两点)
ax.plot([arrow_end[0], y_end[0]],
        [arrow_end[1], arrow_end[1]],
        [arrow_end[2], y_end[2]],
        color='purple', linestyle='dashed', linewidth=LINE_WIDTH)
h=2


# 绘制球形保护区
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = r * np.outer(np.cos(u), np.sin(v)) + drone_position[0]
y = r * np.outer(np.sin(u), np.sin(v)) + drone_position[1]
z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + drone_position[2]
ax.plot_surface(x, y, z, color='gray', alpha=0.8, edgecolor='k', linewidth=0.1, zorder=10)

# 去掉网格和刻度
ax.grid(False)
ax.set_box_aspect([1, 1, 1])  # 设置比例尺为1:1:1
ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 设置图形属性
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-6, 6)
# 设置紧凑型布局
plt.tight_layout()
# 显示图形
plt.show()
