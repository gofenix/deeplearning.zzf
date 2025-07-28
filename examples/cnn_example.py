#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卷积神经网络示例

此示例展示如何使用深度学习ZZF框架构建卷积神经网络（CNN）
用于图像分类任务。使用CIFAR-10数据集。
"""

import deeplearning_zzf as dl
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_cifar10():
    """加载和预处理CIFAR-10数据"""
    print("正在加载CIFAR-10数据集...")
    
    # CIFAR-10类别名称
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', 
                   '狗', '青蛙', '马', '船', '卡车']
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = dl.data.load_cifar10()
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    print(f"类别: {class_names}")
    
    # 数据预处理
    # 1. 归一化像素值到[0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 2. 转换标签为one-hot编码
    y_train = dl.utils.to_categorical(y_train, 10)
    y_test = dl.utils.to_categorical(y_test, 10)
    
    print("数据预处理完成!")
    return (x_train, y_train), (x_test, y_test), class_names


def create_cnn_model():
    """创建CNN模型"""
    print("正在创建CNN模型...")
    
    model = dl.Model(name="cifar10_cnn")
    
    # 第一个卷积块
    model.add_layer(dl.layers.Conv2D(32, (3, 3), activation='relu', 
                                     input_shape=(32, 32, 3)))
    model.add_layer(dl.layers.BatchNormalization())
    model.add_layer(dl.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add_layer(dl.layers.MaxPooling2D((2, 2)))
    model.add_layer(dl.layers.Dropout(0.25))
    
    # 第二个卷积块
    model.add_layer(dl.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add_layer(dl.layers.BatchNormalization())
    model.add_layer(dl.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add_layer(dl.layers.MaxPooling2D((2, 2)))
    model.add_layer(dl.layers.Dropout(0.25))
    
    # 第三个卷积块
    model.add_layer(dl.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add_layer(dl.layers.BatchNormalization())
    model.add_layer(dl.layers.Dropout(0.25))
    
    # 分类器
    model.add_layer(dl.layers.Flatten())
    model.add_layer(dl.layers.Dense(512, activation='relu'))
    model.add_layer(dl.layers.BatchNormalization())
    model.add_layer(dl.layers.Dropout(0.5))
    model.add_layer(dl.layers.Dense(10, activation='softmax'))
    
    print("CNN模型创建完成!")
    return model


def create_data_augmentation():
    """创建数据增强生成器"""
    print("设置数据增强...")
    
    # 训练数据增强
    train_datagen = dl.data.ImageDataGenerator(
        rotation_range=15,          # 随机旋转角度
        width_shift_range=0.1,      # 水平移动
        height_shift_range=0.1,     # 垂直移动
        horizontal_flip=True,       # 水平翻转
        zoom_range=0.1,             # 随机缩放
        fill_mode='nearest'         # 填充模式
    )
    
    # 验证数据不增强
    val_datagen = dl.data.ImageDataGenerator()
    
    return train_datagen, val_datagen


def train_cnn_model(model, train_data, test_data):
    """训练CNN模型"""
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print("正在编译模型...")
    
    # 编译模型
    model.compile(
        optimizer=dl.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 显示模型摘要
    model.summary()
    
    # 创建数据增强
    train_datagen, val_datagen = create_data_augmentation()
    
    # 生成训练和验证数据
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(x_test, y_test, batch_size=32)
    
    # 回调函数
    callbacks = [
        dl.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        dl.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7),
        dl.callbacks.ModelCheckpoint('best_cifar10_model.h5', 
                                   save_best_only=True, monitor='val_accuracy')
    ]
    
    print("开始训练模型...")
    
    # 训练模型
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // 32,
        epochs=50,
        validation_data=val_generator,
        validation_steps=len(x_test) // 32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("训练完成!")
    return history


def visualize_feature_maps(model, x_test, layer_names=None):
    """可视化特征图"""
    if layer_names is None:
        # 选择一些卷积层
        layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3']
    
    # 选择一个测试图像
    test_image = x_test[0:1]  # 形状为 (1, 32, 32, 3)
    
    # 创建特征提取模型
    feature_models = []
    for layer_name in layer_names:
        try:
            layer = model.get_layer(layer_name)
            feature_model = dl.Model(inputs=model.input, 
                                   outputs=layer.output)
            feature_models.append((layer_name, feature_model))
        except:
            print(f"找不到层: {layer_name}")
    
    # 可视化特征图
    fig, axes = plt.subplots(len(feature_models), 8, figsize=(16, 2*len(feature_models)))
    
    for i, (layer_name, feature_model) in enumerate(feature_models):
        features = feature_model.predict(test_image)
        
        for j in range(min(8, features.shape[-1])):
            ax = axes[i, j] if len(feature_models) > 1 else axes[j]
            ax.imshow(features[0, :, :, j], cmap='viridis')
            ax.set_title(f'{layer_name} - 特征{j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_predictions(model, test_data, class_names, num_samples=10):
    """分析预测结果"""
    x_test, y_test = test_data
    
    # 生成预测
    predictions = model.predict(x_test)
    
    # 计算每个类别的准确率
    class_accuracy = []
    for i in range(10):
        class_mask = np.argmax(y_test, axis=1) == i
        if np.sum(class_mask) > 0:
            class_pred = np.argmax(predictions[class_mask], axis=1)
            class_acc = np.mean(class_pred == i)
            class_accuracy.append(class_acc)
        else:
            class_accuracy.append(0.0)
    
    # 显示每个类别的准确率
    print("\n各类别准确率:")
    for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f"{name}: {acc:.3f}")
    
    # 可视化一些预测结果
    plt.figure(figsize=(15, 6))
    
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices[:10]):
        plt.subplot(2, 5, i+1)
        
        # 显示图像
        plt.imshow(x_test[idx])
        
        # 获取真实标签和预测标签
        true_label = np.argmax(y_test[idx])
        pred_label = np.argmax(predictions[idx])
        confidence = np.max(predictions[idx])
        
        # 设置标题（正确预测为绿色，错误为红色）
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'真实: {class_names[true_label]}\n'
                 f'预测: {class_names[pred_label]}\n'
                 f'置信度: {confidence:.2f}', 
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_data, class_names):
    """绘制混淆矩阵"""
    x_test, y_test = test_data
    
    # 生成预测
    predictions = model.predict(x_test)
    
    # 转换为类别索引
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("深度学习ZZF - 卷积神经网络示例 (CIFAR-10)")
    print("=" * 60)
    
    # 设置随机种子
    dl.set_random_seed(42)
    
    try:
        # 1. 加载和预处理数据
        train_data, test_data, class_names = load_and_preprocess_cifar10()
        
        # 2. 创建CNN模型
        model = create_cnn_model()
        
        # 3. 训练模型
        history = train_cnn_model(model, train_data, test_data)
        
        # 4. 分析预测结果
        analyze_predictions(model, test_data, class_names)
        
        # 5. 绘制混淆矩阵
        plot_confusion_matrix(model, test_data, class_names)
        
        # 6. 可视化特征图（如果模型支持）
        try:
            visualize_feature_maps(model, test_data[0])
        except Exception as e:
            print(f"特征图可视化跳过: {e}")
        
        # 7. 绘制训练历史
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "=" * 60)
        print("CNN示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行时出错: {e}")
        raise


if __name__ == "__main__":
    main()