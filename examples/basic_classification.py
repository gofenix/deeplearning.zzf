#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础分类示例

此示例展示如何使用深度学习ZZF框架进行基础的图像分类任务。
使用MNIST手写数字数据集作为示例。
"""

import deeplearning_zzf as dl
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    """加载和预处理MNIST数据"""
    print("正在加载MNIST数据集...")
    
    # 加载MNIST数据
    (x_train, y_train), (x_test, y_test) = dl.data.load_mnist()
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    print(f"类别数量: {len(np.unique(y_train))}")
    
    # 数据预处理
    # 1. 归一化像素值到[0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 2. 重塑数据以添加通道维度
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # 3. 将标签转换为one-hot编码
    y_train = dl.utils.to_categorical(y_train, 10)
    y_test = dl.utils.to_categorical(y_test, 10)
    
    print("数据预处理完成!")
    return (x_train, y_train), (x_test, y_test)


def create_model():
    """创建神经网络模型"""
    print("正在创建模型...")
    
    model = dl.Model(name="mnist_classifier")
    
    # 输入层 - 将28x28的图像展平
    model.add_layer(dl.layers.Flatten(input_shape=(28, 28, 1)))
    
    # 第一个隐藏层
    model.add_layer(dl.layers.Dense(128, activation='relu'))
    model.add_layer(dl.layers.Dropout(0.2))  # 防止过拟合
    
    # 第二个隐藏层
    model.add_layer(dl.layers.Dense(64, activation='relu'))
    model.add_layer(dl.layers.Dropout(0.2))
    
    # 输出层 - 10个类别的分类
    model.add_layer(dl.layers.Dense(10, activation='softmax'))
    
    print("模型创建完成!")
    return model


def compile_and_train_model(model, train_data, test_data):
    """编译和训练模型"""
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print("正在编译模型...")
    
    # 编译模型
    model.compile(
        optimizer='adam',  # Adam优化器
        loss='categorical_crossentropy',  # 多类交叉熵损失
        metrics=['accuracy']  # 监控准确率
    )
    
    # 显示模型摘要
    model.summary()
    
    print("开始训练模型...")
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    print("训练完成!")
    return history


def evaluate_model(model, test_data):
    """评估模型性能"""
    x_test, y_test = test_data
    
    print("正在评估模型...")
    
    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 生成预测
    predictions = model.predict(x_test[:10])
    
    # 显示一些预测结果
    print("\n前10个测试样本的预测结果:")
    for i in range(10):
        true_label = np.argmax(y_test[i])
        pred_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        
        print(f"样本 {i+1}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.3f}")


def plot_training_history(history):
    """绘制训练历史"""
    print("正在绘制训练历史...")
    
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def save_and_load_model(model):
    """保存和加载模型"""
    print("正在保存模型...")
    
    # 保存模型
    model_path = 'mnist_classifier.h5'
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    print("正在加载模型...")
    loaded_model = dl.load_model(model_path)
    print("模型加载成功!")
    
    return loaded_model


def visualize_predictions(model, test_data, num_samples=5):
    """可视化预测结果"""
    x_test, y_test = test_data
    
    # 随机选择一些样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # 获取图像和预测
        image = x_test[idx].reshape(28, 28)
        true_label = np.argmax(y_test[idx])
        
        prediction = model.predict(x_test[idx:idx+1])
        pred_label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 绘制图像
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 50)
    print("深度学习ZZF - 基础分类示例")
    print("=" * 50)
    
    # 设置随机种子以保证结果可重现
    dl.set_random_seed(42)
    
    try:
        # 1. 加载和预处理数据
        train_data, test_data = load_and_preprocess_data()
        
        # 2. 创建模型
        model = create_model()
        
        # 3. 编译和训练模型
        history = compile_and_train_model(model, train_data, test_data)
        
        # 4. 评估模型
        evaluate_model(model, test_data)
        
        # 5. 绘制训练历史
        plot_training_history(history)
        
        # 6. 可视化预测结果
        visualize_predictions(model, test_data)
        
        # 7. 保存和加载模型
        loaded_model = save_and_load_model(model)
        
        print("\n" + "=" * 50)
        print("示例运行完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"运行时出错: {e}")
        raise


if __name__ == "__main__":
    main()