# haiDRL
Code of basic DRL algorithm. 



Try to keep each file executable independently, when one of the algorithms is needed, just need to copy the corresponding file.

### 环境搭建
``` python
conda create -n haiDRL python=3.10
conda activate haiDRL
cd haiDRL
pip install -r requirememts.txt
```
### 单文件结构
每个文件都独立可执行，不依赖于项目中的其他文件，使得需要用到其中一个算法的时候，只要复制一个文件即可。
每个文件主要包括以下4部分内容：
* UTILS：一些算法无关的辅助函数，如路径创建，参数存储等
* NET CONSTRUCTION：网络搭建，定义Actor类, Critic类
* ALGO CONSTRUCTION：算法实现，定义SAC类，PPO类等
* EXECUTE FUNCTION：执行函数
  * get_env_args：获取游戏环境信息，训练无关，只用于了解环境
  * train_gym：执行训练
  * eval_gym：执行模型评估


### 参考资源
[动手学强化学习](https://hrl.boyuai.com/)\
[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)\
[ai_ji](https://github.com/jidiai/ai_lib/tree/master)
