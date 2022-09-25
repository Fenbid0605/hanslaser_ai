# HanslaserAI

大族激光校企合作项目

## 运行前准备

```
pip3 install -r requirements.txt
```

## 目录及文件说明

- data.xlsx 原始数据
- evolution.py 遗传算法
- evolution_test.py 遗传算法测试入口
- config.py 超参数与参数的设置
- dataset.py 数据集对象
- train_model.py 训练入口
- net.py 线性回归网络
- net_test.py 线性回归网络测试入口
- predicted.py 预测对象的数据结构定义
- share_date.py 共享数据的数据结构定义
- result.py 结果记录对象定义
- model.pl 已经训练好的模型文件

- result 前向神经网络训练结果
    - config_log.txt 训练日志
    - loss.png 训练过程 EPOCH 与 Loss 值之间的关系图
    - Train-lab.png 训练集拟合效果图
    - Valid-lab.png 验证集拟合效果图
    - evolution-lab.png 遗传算法拟合效果图

- ui.py GUI 的展示以及回调逻辑
- mainwindow.ui 基于 Qt 的 UI 设计文件
- mainwindow.py 通过 PyUIC 将 UI 文件转化为 Python 对象后的文件

## 运行入口

- 线性回归网络训练入口

```
python3 train_model.py
```

- 线性回归网络测试入口

```
python3 net_test.py
```

- 遗传算法测试入口

```
python3 evolution_test.py
```

- GUI 启动入口

```
python3 ui.py
```