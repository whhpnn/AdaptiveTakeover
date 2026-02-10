import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 导入混合精度训练工具（GPU加速关键）
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib
from tqdm import tqdm

# 全局设置：中文显示、随机种子（保证可复现）
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
torch.manual_seed(42)
np.random.seed(42)
# 设置CUDA随机种子（保证GPU训练可复现）
torch.cuda.manual_seed_all(42)
# 启用CuDNN优化（GPU加速LSTM计算）
torch.backends.cudnn.benchmark = True


# ===================== 1. 轨迹预测专用参数配置 =====================
DATA_DIR = r"/home/kenway/wahaha/PycharmProjects/lstm_tp/overtaking/data"

# 时序参数
INPUT_SEQ_LEN = 40  # 输入序列长度（历史数据行数）
OUTPUT_SEQ_LEN = 20  # 输出序列长度（预测未来行数）
BATCH_SIZE = 32  # 显存充足可改为64/128（如RTX 3060+）
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 关键修改：特征列和目标列定义 ==========
# FEATURE_COLS：2-7列（Excel列索引从0开始，对应索引1,2,3,4,5,6）→ 6维输入
# TARGET_COLS：第二列和第三列（对应索引1,2）→ 2维输出
FEATURE_COLS = [1, 2, 3, 4, 5, 6]  # 第2-7列（6维特征）
TARGET_COLS = [1, 2]  # 目标：第二列和第三列
print(f"使用设备: {DEVICE}")
print(f"CUDA设备名称: {torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else '无'}")
print(f"特征列（Excel第2-7列）: 索引{FEATURE_COLS}")
print(f"预测目标（Excel第2-3列）: 索引{TARGET_COLS}")


# ===================== 2. 轨迹预测专用数据加载与预处理 =====================
def load_excel_trajectory(data_dir):
    all_inputs = []  # 输入序列：(n_samples, 40, 6)
    all_targets = []  # 输出序列：(n_samples, 20, 2)

    # 获取所有Excel文件
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(f"找到{len(excel_files)}个Excel文件")

    for file in tqdm(excel_files, desc="读取轨迹文件"):
        file_path = os.path.join(data_dir, file)
        try:
            # 读取Excel（无表头）
            df = pd.read_excel(file_path, header=None)

            # 数据完整性校验：从第2行（索引1）开始读取所有行
            valid_data = df.iloc[1:].copy()  # 核心修改：第2行开始的所有数据
            if len(valid_data) < INPUT_SEQ_LEN + OUTPUT_SEQ_LEN:
                print(f"警告: 文件{file}有效行数{len(valid_data)} < {INPUT_SEQ_LEN + OUTPUT_SEQ_LEN}，跳过")
                continue
            # 修改列数校验：至少需要7列（索引0-6）→ 适配6维特征
            if valid_data.shape[1] < 7:
                print(f"警告: 文件{file}列数{valid_data.shape[1]} < 7，跳过")
                continue

            # 提取特征和目标（转为numpy数组）
            features = valid_data.iloc[:, FEATURE_COLS].values  # (n_rows, 6)
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


# 加载数据
X, y = load_excel_trajectory(DATA_DIR)
print(f"生成轨迹数据: 输入序列形状={X.shape}, 输出序列形状={y.shape}")
if len(X) == 0:
    print("错误: 没有加载到有效轨迹数据，请检查文件路径和格式")
    exit()

# 数据标准化（输入/输出分开标准化，避免数据泄露）
# 1. 输入特征标准化（6维）
scaler_input = StandardScaler()
n_samples, input_len, n_features = X.shape
X_reshaped = X.reshape(n_samples * input_len, n_features)
X_scaled = scaler_input.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, input_len, n_features)

# 2. 输出目标标准化（第二列和第三列，2维）
scaler_target = StandardScaler()
n_samples, output_len, n_targets = y.shape
y_reshaped = y.reshape(n_samples * output_len, n_targets)
y_scaled = scaler_target.fit_transform(y_reshaped)
y = y_scaled.reshape(n_samples, output_len, n_targets)

# 划分训练集和测试集（shuffle=True，样本独立）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"训练集: X={X_train.shape}, y={y_train.shape}; 测试集: X={X_test.shape}, y={y_test.shape}")


# ===================== 3. 轨迹预测数据集类 =====================
class TrajectoryDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.inputs = input_sequences  # (n_samples, 40, 6)
        self.targets = target_sequences  # (n_samples, 20, 2)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 仅在CPU创建张量（pin_memory需CPU张量）
        input_seq = torch.FloatTensor(self.inputs[idx])
        target_seq = torch.FloatTensor(self.targets[idx])
        return input_seq, target_seq


# 创建数据加载器（pin_memory=True加速CPU→GPU传输）
train_dataset = TrajectoryDataset(X_train, y_train)
test_dataset = TrajectoryDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,  # 锁定CPU内存，加速传输
    num_workers=4  # 多进程加载（适配CPU核心数，避免IO瓶颈）
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=2
)


# ===================== 4. 轨迹预测专用LSTM模型（Seq2Seq结构） =====================
class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=2, output_len=20, dropout=0.2):
        super(LSTMTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len  # 预测步数
        self.output_size = output_size  # 输出维度（第二列和第三列）

        # 编码器LSTM（适配6维输入）
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


# ===================== 5. 模型初始化 =====================
model = LSTMTrajectoryPredictor(
    input_size=len(FEATURE_COLS),  # 6个输入特征（2-7列）
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=len(TARGET_COLS),  # 2个输出（第二列和第三列）
    output_len=OUTPUT_SEQ_LEN
).to(DEVICE)  # 模型转移到GPU/CPU

# 损失函数、优化器、学习率调度器
criterion = nn.MSELoss().to(DEVICE)  # 损失函数转移到设备
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    fused=True  # GPU专属融合优化，加速梯度更新
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=3, factor=0.5, verbose=True
)
# 混合精度训练缩放器（避免float16梯度下溢）
scaler = GradScaler()


# ===================== 6. 模型训练函数 =====================
def train_trajectory_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # ---------------------- 训练阶段 ----------------------
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} 训练"):
            # 数据转移到设备（GPU/CPU），non_blocking=True异步传输（加速）
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # 混合精度前向传播（float16计算，加速且省显存）
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 反向传播与优化
            optimizer.zero_grad()  # 清零梯度
            scaler.scale(loss).backward()  # 梯度缩放（避免下溢）
            scaler.step(optimizer)  # 优化器步骤（自动处理缩放）
            scaler.update()  # 更新缩放因子

            # 累计训练损失（按批次大小加权）
            train_loss += loss.item() * inputs.size(0)

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # ---------------------- 验证阶段 ----------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad(), autocast():  # 验证也用混合精度，加速推理
            for inputs, targets in test_loader:
                # 关键：验证数据也需转移到设备（之前遗漏导致设备不匹配）
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # 计算平均验证损失
        avg_val_loss = val_loss / len(test_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        # 调整学习率（基于验证损失）
        scheduler.step(avg_val_loss)

        # 打印epoch结果（每5轮打印一次，减少IO）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return history


# 启动训练
print("\n开始轨迹预测模型训练...")
history = train_trajectory_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS
)


# ===================== 7. 训练过程可视化 =====================
plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label='训练损失（MSE）', linewidth=1.5)
plt.plot(history['val_loss'], label='验证损失（MSE）', linewidth=1.5)
plt.title('轨迹预测模型损失变化（GPU加速）')
plt.xlabel('Epoch')
plt.ylabel('均方误差（MSE）')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('trajectory_training_loss.png', dpi=300)
plt.show()


# ===================== 8. 模型评估 =====================
def evaluate_trajectory_model(model, test_loader, scaler_target):
    model.eval()
    all_preds = []  # 预测序列（标准化后）
    all_targets = []  # 真实序列（标准化后）

    with torch.no_grad(), autocast():
        for inputs, targets in test_loader:
            # 数据转移到设备（关键：避免设备不匹配）
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            # 异步转移到CPU（不阻塞GPU）
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 逆标准化：恢复到原始数据尺度（重要！指标才具物理意义）
    all_preds = np.array(all_preds)  # (n_test, 20, 2)
    all_targets = np.array(all_targets)  # (n_test, 20, 2)
    n_test, out_len, n_targets = all_preds.shape

    # 展平后逆标准化，再恢复3D形状
    preds_original = scaler_target.inverse_transform(
        all_preds.reshape(-1, n_targets)
    ).reshape(n_test, out_len, n_targets)
    targets_original = scaler_target.inverse_transform(
        all_targets.reshape(-1, n_targets)
    ).reshape(n_test, out_len, n_targets)

    # 计算评估指标
    # 1. 整体指标
    overall_mse = mean_squared_error(
        targets_original.reshape(-1, n_targets),
        preds_original.reshape(-1, n_targets)
    )
    overall_mae = mean_absolute_error(
        targets_original.reshape(-1, n_targets),
        preds_original.reshape(-1, n_targets)
    )

    # 2. 逐步指标（每一步的预测精度）
    step_mse = [
        mean_squared_error(targets_original[:, step, :], preds_original[:, step, :])
        for step in range(out_len)
    ]

    # 打印结果
    print("\n=== 轨迹预测模型评估结果 ===")
    print(f"整体MSE（均方误差）: {overall_mse:.6f}")
    print(f"整体MAE（平均绝对误差）: {overall_mae:.6f}")
    print("\n未来20步逐步MSE:")
    for i in range(4):  # 每5步一行，便于查看
        step_range = range(i*5, (i+1)*5)
        print(" | ".join([f"第{step+1}步: {step_mse[step]:.6f}" for step in step_range]))

    # 可视化：随机选5个样本对比真实vs预测轨迹
    plt.figure(figsize=(15, 10))
    sample_indices = np.random.choice(len(preds_original), 5, replace=False)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # 鲜明配色

    for idx, sample_idx in enumerate(sample_indices):
        # 提取单个样本的预测和真实值（第二列和第三列）
        true_col1 = targets_original[sample_idx, :, 0]  # 第二列真实值
        true_col2 = targets_original[sample_idx, :, 1]  # 第三列真实值
        pred_col1 = preds_original[sample_idx, :, 0]    # 第二列预测值
        pred_col2 = preds_original[sample_idx, :, 1]    # 第三列预测值

        # 子图绘制：第二列对比
        plt.subplot(2, 5, idx+1)
        plt.plot(true_col1, marker='o', markersize=4, label='真实第二列',
                 color=colors[idx], alpha=0.8, linewidth=2)
        plt.plot(pred_col1, marker='s', markersize=3, label='预测第二列',
                 color='#7f8c8d', linestyle='--', alpha=0.8, linewidth=2)
        plt.title(f'样本{sample_idx+1}：第二列对比', fontsize=10)
        plt.xlabel('预测步数', fontsize=9)
        plt.ylabel('数值', fontsize=9)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)

        # 子图绘制：第三列对比
        plt.subplot(2, 5, idx+6)
        plt.plot(true_col2, marker='o', markersize=4, label='真实第三列',
                 color=colors[idx], alpha=0.8, linewidth=2)
        plt.plot(pred_col2, marker='s', markersize=3, label='预测第三列',
                 color='#7f8c8d', linestyle='--', alpha=0.8, linewidth=2)
        plt.title(f'样本{sample_idx+1}：第三列对比', fontsize=10)
        plt.xlabel('预测步数', fontsize=9)
        plt.ylabel('数值', fontsize=9)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300)
    plt.show()

    return preds_original, targets_original


# 执行评估
print("\n开始测试集评估...")
preds_original, targets_original = evaluate_trajectory_model(model, test_loader, scaler_target)


# ===================== 9. 保存模型与标准化器 =====================
# 保存模型参数（仅保存状态字典，轻量且灵活）
torch.save(model.state_dict(), 'lstm_trajectory_predictor_gpu.pth')
# 保存标准化器（预测时需复用相同的标准化逻辑）
import joblib
joblib.dump(scaler_input, 'scaler_trajectory_input.pkl')
joblib.dump(scaler_target, 'scaler_trajectory_target.pkl')

print("\n文件保存完成：")
print("1. 模型参数: lstm_trajectory_predictor_gpu.pth")
print("2. 输入特征标准化器: scaler_trajectory_input.pkl")
print("3. 目标列标准化器: scaler_trajectory_target.pkl")