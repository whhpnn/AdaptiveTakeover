#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import joblib

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

from scipy.stats import genextreme
import queue
from scipy.optimize import minimize


import pandas as pd
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    from collections import deque
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================



DATA_DIR = r"D:\\software\CARLA_0.9.11\WindowsNoEditor\wrs_demo\curve\data" # 替换为你的路径
# 2.2 时序与特征参数（和训练代码完全一致，否则标准化器不匹配）
INPUT_SEQ_LEN = 40    # 输入序列长度（历史40步）
OUTPUT_SEQ_LEN = 20   # 输出序列长度（预测20步）
FEATURE_COLS = [1, 2, 3, 4, 5]  # 输入特征列
TARGET_COLS = [1, 2]  # 目标列
# 2.3 模型参数（和训练代码完全一致）
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MODEL_PATH = r"D:\\software\CARLA_0.9.11\WindowsNoEditor\wrs_demo\curve\lstm_trajectory_predictor_gpu.pth"  # 训练好的模型权重路径
# 2.4 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


# ===================== 3. 生成标准化器（部署阶段核心：不保存文件，直接拟合） =====================
def load_excel_trajectory(data_dir):
    all_inputs = []  # 输入序列：(n_samples, 40, 5)
    all_targets = []  # 输出序列：(n_samples, 20, 2)

    # 获取所有Excel文件
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(f"找到{len(excel_files)}个Excel文件")

    for file in tqdm(excel_files, desc="读取轨迹文件"):
        file_path = os.path.join(data_dir, file)
        try:
            # 读取Excel（无表头）
            df = pd.read_excel(file_path, header=None)

            # 数据完整性校验
            valid_data = df.iloc[100:480].copy()  # 保留100-480行有效数据
            if len(valid_data) < INPUT_SEQ_LEN + OUTPUT_SEQ_LEN:
                print(f"警告: 文件{file}有效行数{len(valid_data)} < {INPUT_SEQ_LEN + OUTPUT_SEQ_LEN}，跳过")
                continue
            # 修改列数校验：至少需要6列（索引0-5）
            if valid_data.shape[1] < 6:
                print(f"警告: 文件{file}列数{valid_data.shape[1]} < 6，跳过")
                continue

            # 提取特征和目标（转为numpy数组）
            features = valid_data.iloc[:, FEATURE_COLS].values  # (n_rows, 5)
            targets = valid_data.iloc[:, TARGET_COLS].values  # (n_rows, 2)

            # 滑动窗口生成输入输出对
            max_start_idx = len(valid_data) - INPUT_SEQ_LEN - OUTPUT_SEQ_LEN + 1
            for i in range(max_start_idx):
                input_seq = features[i:i + INPUT_SEQ_LEN]
                target_seq = targets[i + INPUT_SEQ_LEN:i + INPUT_SEQ_LEN + OUTPUT_SEQ_LEN]
                all_inputs.append(input_seq)
                all_targets.append(target_seq)

        except Exception as e:
            print(f"读取文件{file}出错: {str(e)}")

    return np.array(all_inputs), np.array(all_targets)

def generate_scalers():
    """拟合并返回scaler_input和scaler_xy（不保存文件）"""
    # 加载训练数据
    X, y = load_excel_trajectory(DATA_DIR)
    print(f"训练数据加载完成：输入序列形状={X.shape}, 输出序列形状={y.shape}")
    if len(X) == 0:
        print("错误：没有有效训练数据，无法生成标准化器！")
        sys.exit()

    # 1. 拟合输入特征标准化器（和训练一致的逻辑）
    scaler_input = StandardScaler()
    n_samples, input_len, n_features = X.shape
    X_reshaped = X.reshape(n_samples * input_len, n_features)  # 展平为2D：(样本数×步长, 特征数)
    scaler_input.fit(X_reshaped)  # 用所有训练数据拟合

    # 2. 拟合XY坐标标准化器（和训练一致的逻辑）
    scaler_xy = StandardScaler()
    n_samples, output_len, n_targets = y.shape
    y_reshaped = y.reshape(n_samples * output_len, n_targets)  # 展平为2D：(样本数×步长, 目标数)
    scaler_xy.fit(y_reshaped)  # 用所有训练数据拟合

    print("标准化器生成完成（未保存文件，直接用于预测）\n")
    return scaler_input, scaler_xy







class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=2, output_len=20, dropout=0.2):
        super(LSTMTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len  # 预测步数
        self.output_size = output_size  # 输出维度（第二列和第三列）

        # 编码器LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层（映射到目标输出）
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 确保输入与模型在同一设备（关键：解决设备不匹配）
        x = x.to(self.encoder_lstm.weight_ih_l0.device)
        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态（与输入同设备）
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # 编码器前向传播
        encoder_out, (hn, cn) = self.encoder_lstm(x, (h0, c0))
        decoder_input = encoder_out[:, -1:, :]  # 取编码器最后一步输出

        # 解码器逐步生成预测序列
        predictions = []
        for _ in range(self.output_len):
            decoder_out, (hn, cn) = self.decoder_lstm(decoder_input, (hn, cn))
            decoder_out = self.dropout(decoder_out)
            pred = self.fc(decoder_out)  # 映射到目标输出
            predictions.append(pred)
            decoder_input = decoder_out  # 教师强制：下一步输入=当前输出

        # 拼接所有预测步（batch_size, 20, 2）
        predictions = torch.cat(predictions, dim=1)
        return predictions


MODEL_PATH = "lstm_trajectory_predictor_gpu.pth"
SCALER_INPUT_PATH = "scaler_trajectory_input.pkl"
SCALER_XY_PATH = "scaler_trajectory_xy.pkl"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMTrajectoryPredictor(
    input_size=len(FEATURE_COLS),  # 5个输入特征（2-6列）
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=len(TARGET_COLS),  # 2个输出（第二列和第三列）
    output_len=OUTPUT_SEQ_LEN
).to(DEVICE)  # 模型转移到GPU/CPU

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


print("正在生成标准化器...")
scaler_input, scaler_xy = generate_scalers()  # 调用前面定义的生成函数
print("标准化器生成完成")










def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04_opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        # Toggle all buildings off
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.All)
        # Toggle all buildings on
        #self.world.load_map_layer(carla.MapLayer.Buildings)
        self.actor_role_name = args.rolename
        self.realcp = []
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        self.waypoints = []
        self.map = self.world.get_map()


    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # 设置采样频率和同步模式
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)




        # Get a random blueprint.
        blueprint = self.world.get_blueprint_library().filter('model3')[0]
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            #color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', '0, 0, 0')
        if blueprint.has_attribute('driver_id'):
            #driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', 'vehicle.lincoln.mkz2017')
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)

            spawn_point = carla.Transform(carla.Location(x=402.478240966797, y=-120, z=1),
                                          carla.Rotation(yaw=-89.314423))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
            self.player.enable_constant_velocity(carla.Vector3D(x=5, y=0.0, z=0.0))

            location = carla.Location(x=381.392028808594, y=-312.0944519, z=1)
            # 大约八十米处任务结束，设置终点标记
            self.world.debug.draw_point(
                location,
                size=0.2,
                color=carla.Color(255, 0, 0),
                life_time=1000.0
            )
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display,delay,TOR):
        self.camera_manager.render(display,delay,TOR)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self.DEVICE = DEVICE  # 直接使用全局设备配
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.model = model  # 加载好的LSTM模型
        self.scaler_input = scaler_input  # 部署时生成的标准化器
        self.scaler_xy = scaler_xy
        # 定义设备（与训练时一致，优先GPU）
        self.DEVICE = DEVICE
        self.statelist=[]
        self.takeover=0
        self.TOR=0
        self.timestamp=0
        self.xy=[]
        self.has_printed_boundary = False
        self.has_printed_takeover = False
        self.has_end = False
        self.qq = deque()
        self.steer=[]
        self.P=[]
        self.t_demand=0
        self.t_real=0
        self.k = 1
        self.ulast = 0



        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))




    def parse_events(self, client, world, clock, delay):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time(), world, delay)
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds, world, delay):
        fbdelay=10
        self.timestamp = self.timestamp + 1
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        T = jsInputs[self._steer_idx]
        K1 = 1 # 0.55
        steerCmds = K1 * math.tan(1.1 * T)

        if len(self.qq) < delay:
            for i in range(delay - len(self.qq)):
                element = []
                element.append((None, self.timestamp))
                self.qq.append(element)
            element = []
            element.append(( steerCmds, self.timestamp))
            self.qq.append(element)
        elif len(self.qq) > delay:
            element = self.qq[delay]
            if element[0] != None:
                element.append(( steerCmds, self.timestamp))
            else:
                element = []
                element.append(( steerCmds, self.timestamp))
            self.qq[delay] = element
        else:
            element = []
            element.append(( steerCmds, self.timestamp))
            self.qq.append(element)


        v = world.player.get_velocity()
        v0 = float(np.sqrt(np.power(v.x, 2) + np.power(v.y, 2)))
        x = world.player.get_location().x
        y = world.player.get_location().y
        transform = world.player.get_transform().rotation.yaw
        the = float(90 - transform) * np.pi / 180
        arr = [x,y, transform, v0, None]
        self.statelist.append(arr)

        steerCmd = self.qq.popleft()
        for i in range(len(steerCmd)):
            if steerCmd[i][0] != None:
                if steerCmd[i][0] != 0:
                    self.takeover = 1
                self._steer_cache = round(steerCmd[i][0], 1)
                self._steer_cache = min(1, max(-1, self._steer_cache))
                if (steerCmd[i][1] - fbdelay) != None and steerCmd[i][1] - fbdelay > 0 and self.takeover == 1:
                    self.statelist[steerCmd[i][1] - 1 - fbdelay][4] = self._steer_cache



        P_real = self.calculate_distance_to_curve(x, y, world.realcp)
        self.P.append(P_real)
        if P_real > 0.6:
            if not self.has_printed_boundary:
               self.t_demand = self.timestamp
               self.has_printed_boundary = True



        if self.takeover == 0:
            u_lanekeeping = self.lanekeeping(x, y, the, v0)
            steercache_lanekeeping = np.clip(u_lanekeeping / 1.2217, -1, 1)
            self._control.steer =  steercache_lanekeeping
            self.statelist[len(self.statelist)-1][4] = steercache_lanekeeping
            x_predicted = x
            y_predicted = y
            the_predicted = the
            for i in range(16):
                the_predicted = the_predicted + 0.05 * v0 * np.tan(self.lanekeeping(x_predicted, y_predicted, the_predicted, v0)) / 3.6
                x_predicted = x_predicted + v0 * 0.05 * np.sin(the_predicted)  # 系统运动学
                y_predicted = y_predicted + v0 * 0.05 * np.cos(the_predicted)
                P_predicted = self.calculate_distance_to_curve( x_predicted,  y_predicted, world.realcp)
                if P_predicted > 0.6:
                    self.TOR = 1
                    break
            #接管开始
        else:

            if not self.has_printed_takeover:
                self.t_real = self.timestamp
                self.has_printed_takeover = True
            self._control.steer = round(self._steer_cache, 1)

            self.xy.append([x,y])

            if len(self.statelist) > (40+16):
                tf = len(self.statelist) - 1 - 16
                for i in range(16):
                    current_index = tf + i + 1
                    # 检查当前位置的d值是否为None
                    if self.statelist[current_index][4] is None:
                        current_index = current_index - 1
                        break

            realdelay = self.timestamp - current_index - 1
            slidewindow = self.statelist[current_index - 39 :current_index + 1]
            input_data = np.array(slidewindow, dtype=np.float32)
            output_data = self.predict_trajectory(input_data)
            intended_x = output_data[realdelay:realdelay+4, 0]
            intended_y = output_data[realdelay:realdelay+4, 1]
            umpc = self.mpccontrol(x, y, v0, the, intended_x,intended_y, self.ulast)
            steer_cachempc =   np.clip( - umpc / 1.2217, -1, 1)
            orginx=input_data[39,0]
            orginy=input_data[39,1]
            Dk= np.sqrt((x -orginx) ** 2 + (y - orginy) ** 2)
            P1=0.0015
            P2=20
            if P_real > 3:
                detk = -1
            else:
                detk=-P1*(P2-Dk)
            self.k = self.k + detk
            if self.k < 0:
                self.k = 0
            u_lanekeeping = self.lanekeeping(x, y, the, v0)
            steercache_lanekeeping = np.clip(u_lanekeeping / 1.2217, -1, 1)
            self._control.steer = (1 - self.k) * steer_cachempc+ self.k * steercache_lanekeeping
            self.ulast = self._control.steer * 1.2217

            physics_control = world.player.get_physics_control()  # 获取车辆物理参数
            max_steer_angle = physics_control.wheels[0].max_steer_angle  # 前轮的最大转向角（弧度）
            self.steer.append(abs(max_steer_angle * self._control.steer))



    def mpccontrol(self, x0, y0, v0, the0, intended_x, intended_y, ulast0):
        u = [0, 0, 0, 0]  # 控制输入初始化

        def cost_function(u):
            ulast = ulast0
            x = x0
            y = y0
            the = the0
            cost = 0.0
            for i in range(4):
                the = the + 0.05 * v0 * np.tan(u[i]) / 3.6
                x = x + v0 * 0.05 * np.sin(the)  # 系统动力学
                y = y + v0 * 0.05 * np.cos(the)
                detau = u[i] - ulast
                cost += 4* (x - intended_x[i]) ** 2 + 4* (y - intended_y[i]) ** 2 + 2 * u[i] ** 2 + 2*detau**2  # 代价函数
                ulast = u[i]
            return cost

        def con(args):
            x0, v0, the0 = args
            cons = ({'type': 'ineq', 'fun': lambda u: -u[0] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: u[0] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: -u[1] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: u[1] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: -u[2] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: u[2] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: -u[3] + 0.5}, \
                    {'type': 'ineq', 'fun': lambda u: u[3] + 0.5})
            return cons

        # 优化求解
        args2 = (x0, v0, the0)
        cons = con(args2)
        result = minimize(cost_function, x0=u, constraints=cons)
        return result.x[0]  # 返回最优控制输入

    def calculate_tracking_mse(self,actual_xy, centerline_cp):
        """
        actual_xy: list of (x, y), e.g. [(x1, y1), (x2, y2), ...]
        centerline_cp: longer list of (x, y), e.g. [(cx1, cy1), (cx2, cy2), ...]
        return: float MSE
        """
        actual_xy = np.array(actual_xy)
        centerline_cp = np.array(centerline_cp)

        mse_sum = 0
        for point in actual_xy:
            # 计算当前点与所有中心线点之间的欧氏距离
            distances = np.linalg.norm(centerline_cp - point, axis=1)
            min_dist = np.min(distances)
            mse_sum += min_dist ** 2

        mse = mse_sum / len(actual_xy)
        return mse

    def calculate_distance_to_curve(self,x, y, statelist):
        """
        计算点 (x, y) 到曲线（由 statelist 中的点构成）的最短距离。
        :param x: 目标点的 x 坐标
        :param y: 目标点的 y 坐标
        :param statelist: 包含多个点的列表，每个元素是一个[x, y]的坐标，构成曲线
        :return: 目标点到曲线的最短距离
        """

        def point_to_segment_distance(px, py, ax, ay, bx, by):
            """
            计算点(px, py)到线段(ab)的最短距离。
            """
            # 向量 AB 和 AP
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay

            # 向量 AB 的长度的平方
            ab_square = abx ** 2 + aby ** 2

            # 投影系数 t
            if ab_square != 0:  # 防止除以0
                t = (apx * abx + apy * aby) / ab_square
            else:
                t = 0

            # 计算最近点的坐标
            if t < 0:  # 最近点在 A 之前
                nearest_x, nearest_y = ax, ay
            elif t > 1:  # 最近点在 B 之后
                nearest_x, nearest_y = bx, by
            else:  # 最近点在线段 AB 上
                nearest_x = ax + t * abx
                nearest_y = ay + t * aby

            # 计算点到最近点的距离
            return np.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)

        # 计算目标点到每个线段的距离，并找到最小的那个
        min_distance = float('inf')
        for i in range(len(statelist) - 1):
            x1, y1 = statelist[i]
            x2, y2 = statelist[i + 1]
            distance = point_to_segment_distance(x, y, x1, y1, x2, y2)
            min_distance = min(min_distance, distance)

        return min_distance


    def lanekeeping(self, x0, y0, the0, v0):
        """
        车道保持控制器：基于拟合直线计算横向偏差，输出控制输入
        :param x0: 车辆当前x坐标
        :param y0: 车辆当前y坐标
        :param the0: 车辆当前航向角（单位：弧度）
        :param v0: 车辆当前速度（单位：m/s）
        :return: 自主控制器的控制输入（转向角/控制量，单位：弧度）
        """


        # ---------------------- 横向偏差(ey)计算 ----------------------
        # 点(x0,y0)到直线 ax + by + c = 0 的距离公式：|ax0 + by0 + c| / √(a²+b²)
        # 转换拟合直线为标准式：98.085443x + y - 40388.850034 = 0
        a = -95.7132 # 标准式系数a
        c = 38402.2656  # 标准式系数c
        b = -1
        ey = (a * x0 + b * y0 + c) / np.sqrt(a ** 2 + 1 ** 2)
        u1 = (the0- 178 * np.pi / 180 ) + np.arctan(1.8 * ey / v0)

        return u1  # 返回自主控制器的控制输入


    # ===================== 4. 预测函数（核心） =====================
    def predict_trajectory(self, input_sequence):
        """用LSTM模型预测轨迹（使用部署时生成的scaler）"""
        # 输入格式校验（确保和训练一致：(40,7)）
        if input_sequence.shape != (INPUT_SEQ_LEN, len(FEATURE_COLS)):
            raise ValueError(f"输入形状需为({INPUT_SEQ_LEN}, {len(FEATURE_COLS)})，实际为{input_sequence.shape}")

        # 标准化输入（用部署时生成的scaler_input）
        input_scaled = self.scaler_input.transform(
            input_sequence.reshape(-1, len(FEATURE_COLS))
        ).reshape(1, INPUT_SEQ_LEN, len(FEATURE_COLS))

        # 模型预测
        input_tensor = torch.FloatTensor(input_scaled).to(self.DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred_scaled = self.model(input_tensor)

        # 逆标准化（用部署时生成的scaler_xy）
        pred_original = self.scaler_xy.inverse_transform(
            pred_scaled.cpu().numpy().reshape(-1, len(TARGET_COLS))
        ).reshape(OUTPUT_SEQ_LEN, len(TARGET_COLS))

        return pred_original




    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
       # self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.s = queue.Queue(10000)
        self.tor = queue.Queue(10000)
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display, delay,TOR):
        if self.surface is not None:
            if self.tor.qsize() < delay:
                for i in range(delay - self.tor.qsize()):
                    self.tor.put(None)
            elif self.tor.qsize() > delay:
                queuetor2 = queue.Queue(100)
                for i in range(self.tor.qsize()):
                    queuetor2.put(self.tor.get())
                for i in range(delay):
                    self.tor.put(queuetor2.get())

            self.tor.put(TOR)
            TOR = self.tor.get()
            if TOR ==1:
                self.hud.notification("Please Takeover!", seconds=4.0)


            if self.s.qsize() < delay:
                for i in range(delay - self.s.qsize()):
                    self.s.put(None)
            elif self.s.qsize() > delay:
                queue2 = queue.Queue(100)
                for i in range(self.s.qsize()):
                    queue2.put(self.s.get())
                for i in range(delay):
                    self.s.put(queue2.get())
            self.s.put(self.surface)
            A = self.s.get()

            if A != None:
                display.blit(A, (0, 0))
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None


    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()

        datacontrol = genextreme.rvs(c=0.29, loc=0.2, scale=0.009, size=100000)
        datafeedback = genextreme.rvs(c=0.58, loc=0.4, scale=0.018, size=100000)

        file_path = 'D:\\software\\CARLA_0.9.11\\WindowsNoEditor\\wrs_demo\\curve\\\cp.xlsx'

        df = pd.read_excel(file_path, skiprows=1, nrows=1489)  # 跳过第1行，读取1489行

        # 提取第二列和第三列数据，分别作为 x 和 y
        x = df.iloc[:, 1]  # 第二列是 x
        y = df.iloc[:, 2]  # 第三列是 y

        # 将第二列和第三列的值组合成一个列表，每个元素是 [x, y]
        world.realcp = list(zip(x, y))

        while True:
            delaycontrol = int(1000 * np.random.choice(datacontrol) / 50)
            delayfeedback = 10
            world.world.tick()
            clock.tick_busy_loop(20)
            if controller.parse_events(client, world, clock,delaycontrol):
                return
            world.tick(clock)
            world.render(display, delayfeedback, controller.TOR)
            pygame.display.flip()

    finally:
        settings = client.get_world().get_settings()
        settings.synchronous_mode = False
        client.get_world().apply_settings(settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
