# HanslaserAI

大族激光校企合作项目

## 目录及文件说明
- data 存放原始数据
- evolution_algorithm 遗传算法
  - main.py 遗传算法

- forward 前向神经网络
  - config.py 超参数配置
  - dataset.py 数据集对象
  - main.py 训练入口
  - net.py 线性回归网络
  - test.py 测试入口


- result 前向神经网络训练结果
    - config_log.txt 训练日志
    - loss.png 训练过程 EPOCH 与 Loss 值之间的关系图
    - Train-lab.png 训练集拟合效果图
    - Valid-lab.png 验证集拟合效果图

- ui.py GUI 的展示以及回调逻辑
- mainwindow.ui 基于 Qt 的 UI 设计文件
- mainwindow.py 通过 PyUIC 将 UI 文件转化为 Python 对象后的文件

