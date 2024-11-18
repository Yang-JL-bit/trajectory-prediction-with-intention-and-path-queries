## 调参记录
——————

### 2024-11-12
使用tpe对**完整模型**进行了调参
搜索空间为
```python
space = {
    "hidden_size": hp.choice("hidden_size", [16, 32, 64, 128]),
    "num_layers": hp.choice("num_layers", [1, 2, 3]),
    "head_num": hp.choice("head_num", [1, 2, 4, 8]),
    "lr": hp.choice("lr", [0.0001, 0.001, 0.01, 0.1]),
    "epoch": hp.choice("epoch", [50, 100, 150, 200]),
    "batch_size": hp.choice("batch_size", [32, 64, 128, 256, 512]),
    "patience": hp.randint("patience", 10)
}
```

调参结果： 
```python
"best_params": {
            "batch_size": "4",
            "epoch": "0",
            "head_num": "3",
            "hidden_size": "0",
            "lr": "1",
            "num_layers": "1",
            "patience": "5"
        }
```

**训练结果**：测试集精度为：==99.42644551232367%==



---

## 2024-11-15

对不添加feature_weighting的模型进行了调参

搜索空间同上

调参结果

~~~	python
"best_params": {
            "batch_size": "2",
            "epoch": "1",
            "head_num": "3",
            "hidden_size": "1",
            "lr": "0",
            "num_layers": "1",
            "patience": "4"
        }

~~~

**训练结果**： 测试集精度为： ==98.89%==

