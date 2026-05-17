import copy
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.nn import GCNConv
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

seed = random.SystemRandom().randrange(2**32)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
# dataset = Flickr(root='data/Flickr', transform=NormalizeFeatures())
# dataset = Planetoid(root='data', name='PubMed', transform=NormalizeFeatures())
dataset = Coauthor(root='data', name='Physics', transform=NormalizeFeatures())
data = dataset[0]  # 获取图数据


def has_mask(data, name):
    return hasattr(data, name) and getattr(data, name) is not None


def ensure_train_val_test_masks(data):
    mask_names = ('train_mask', 'val_mask', 'test_mask')
    if all(has_mask(data, name) for name in mask_names):
        return data

    print('数据集未提供 train_mask/val_mask/test_mask，随机划分为 60%/20%/20%。')
    splitter = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    return splitter(data)


data = ensure_train_val_test_masks(data)

# 打印数据集基本信息
print(f'数据集: {dataset}')
print(f'图节点数: {data.num_nodes}')
print(f'图边数: {data.num_edges}')
print(f'节点特征维度: {dataset.num_features}')
print(f'类别数: {dataset.num_classes}')
print(f'训练集节点数: {data.train_mask.sum().item()}')
print(f'验证集节点数: {data.val_mask.sum().item()}')
print(f'测试集节点数: {data.test_mask.sum().item()}')
print(f'训练设备: {device}')
print(f'随机种子: {seed}')

data = data.to(device)

# 定义使用GCNConv的两层GCN模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # 第一层GCN
        self.conv2 = GCNConv(hidden_channels, out_channels)  # 第二层GCN
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # 第一层: A F M1，然后ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层: A (ReLU(A F M1)) M2
        x = self.conv2(x, edge_index)
        # 返回log_softmax以便使用NLLLoss
        return F.log_softmax(x, dim=1)

# 模型参数
in_channels = dataset.num_features
hidden_channels = 16
out_channels = dataset.num_classes
dropout = 0.4
learning_rate = 0.01
weight_decay = 5e-4
epochs = 1000
patience = 150

# 初始化模型、优化器和损失函数
model = GCN(in_channels, hidden_channels, out_channels, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=40,
)
criterion = nn.NLLLoss()  # 负对数似然损失，与log_softmax配合

# 训练函数
def train():
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(data.x, data.edge_index)  # 前向传播
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 仅训练节点计算损失
    loss.backward()
    optimizer.step()
    return loss.item()

# 评估函数
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # 获取预测类别
        correct = (pred[mask] == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        
        # 返回准确率和预测结果
        return acc, out

def evaluate_all():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        return train_acc, val_acc, test_acc, out

def get_dataset_output_name(dataset):
    return getattr(dataset, 'name', dataset.__class__.__name__).lower()

def save_matrix(path, matrix, fmt='%.10g'):
    np.savetxt(path, matrix, fmt=fmt)

def save_bidirectional_edges(path, edge_index):
    edge_array = edge_index.detach().cpu().t().numpy()
    edge_pairs = set()
    for src, dst in edge_array:
        src = int(src)
        dst = int(dst)
        edge_pairs.add((src, dst))
        edge_pairs.add((dst, src))

    with open(path, 'w') as file:
        for src, dst in sorted(edge_pairs):
            file.write(f'{src} {dst}\n')

def remap_labels_to_zero(labels):
    labels = labels.detach().cpu().numpy()
    unique_labels = sorted(np.unique(labels).tolist())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)

def export_dataset_and_weights(data, model, dataset):
    output_dir = Path('dataset') / get_dataset_output_name(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_bidirectional_edges(output_dir / 'edges.txt', data.edge_index)
    save_matrix(output_dir / 'features.txt', data.x.detach().cpu().numpy(), fmt='%.10g')
    save_matrix(output_dir / 'labels.txt', remap_labels_to_zero(data.y), fmt='%d')
    save_matrix(output_dir / 'w1.txt', model.conv1.lin.weight.detach().cpu().numpy(), fmt='%.10g')
    save_matrix(output_dir / 'w2.txt', model.conv2.lin.weight.detach().cpu().numpy(), fmt='%.10g')

    print(f'\n数据集与模型权重已导出到: {output_dir}')
    print('包含文件: edges.txt, features.txt, labels.txt, w1.txt, w2.txt')

# 训练循环
train_losses = []
train_accs = []
val_accs = []
test_accs = []
best_val_acc = 0.0
best_epoch = 0
best_state_dict = copy.deepcopy(model.state_dict())
best_metrics = (0.0, 0.0, 0.0)

for epoch in range(1, epochs + 1):
    loss = train()
    train_acc, val_acc, test_acc, _ = evaluate_all()
    scheduler.step(val_acc)
    
    train_losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_state_dict = copy.deepcopy(model.state_dict())
        best_metrics = (train_acc, val_acc, test_acc)
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.5f}')

    if epoch - best_epoch >= patience:
        print(f'Early stopping at epoch {epoch:03d}; best epoch was {best_epoch:03d}.')
        break

model.load_state_dict(best_state_dict)

# 最终测试
train_acc, train_out = evaluate(data.train_mask)
val_acc, val_out = evaluate(data.val_mask)
test_acc, test_out = evaluate(data.test_mask)

print(f'\n最终结果:')
print(f'最佳验证轮数: {best_epoch}')
print(f'最佳验证轮数对应准确率: Train {best_metrics[0]:.4f}, '
      f'Val {best_metrics[1]:.4f}, Test {best_metrics[2]:.4f}')
print(f'训练集准确率: {train_acc:.4f}')
print(f'验证集准确率: {val_acc:.4f}')
print(f'测试集准确率: {test_acc:.4f}')

export_dataset_and_weights(data, model, dataset)

# 可视化训练过程
def plot_training_history(train_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(train_accs, label='训练准确率', color='green')
    ax2.plot(val_accs, label='验证准确率', color='orange')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('准确率')
    ax2.set_title('训练和验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gcn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

# 绘制训练历史
# plot_training_history(train_losses, train_accs, val_accs)

# 可视化部分节点的预测结果
def visualize_predictions(data, model, num_nodes=20):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
    # 选择前num_nodes个节点展示真实标签和预测标签
    nodes = list(range(min(num_nodes, data.num_nodes)))
    true_labels = data.y[nodes].cpu().numpy()
    pred_labels = pred[nodes].cpu().numpy()
    
    print(f'\n前{len(nodes)}个节点的预测结果:')
    print('节点ID | 真实类别 | 预测类别 | 是否正确')
    print('-' * 40)
    correct_count = 0
    for i, node_id in enumerate(nodes):
        is_correct = true_labels[i] == pred_labels[i]
        if is_correct:
            correct_count += 1
        print(f'{node_id:6d} | {true_labels[i]:8d} | {pred_labels[i]:8d} | {is_correct}')
    
    print(f'\n准确率: {correct_count}/{len(nodes)} = {correct_count/len(nodes):.2%}')

# 展示部分节点的预测结果
# visualize_predictions(data, model, num_nodes=20)

# 保存模型
# torch.save(model.state_dict(), 'gcn_model.pth')
# print('\n模型已保存到 gcn_model.pth')
