# 可视化 Q 学习

## 强化学习(RL)
研究机器人如何从成功与失败、回报与惩罚中进行学习。

![image](https://github.com/foamliu/Visualize-Q-Learning/raw/master/images/RL.png)

强化学习的强大能来源于两个方面：使用样本来优化行为，使用函数近似来描述复杂的环境。它们使得强化学习可以使用在以下的复杂环境中：
- 模型的环境未知，且解析解不存在；
- 仅仅给出环境的模拟模型（模拟优化方法的问题）
- 从环境中获取信息的唯一办法是和它互动。

## 依赖库

- [Gym](https://gym.openai.com/)

## OpenAI Gym

OpenAI成立于2015年底，目标是“建立安全的人工通用智能(AGI)，并确保其惠及大众”。

Gym是为测试和开发RL算法而设计的环境/任务的集合。它让用户不必再创建复杂的环境。

OpenAI对机器学习世界的一个主要贡献是开发了Gym和Universe软件平台。

Gym用Python编写，它有很多的环境，比如机器人模拟或Atari 游戏。它还提供了一个在线排行榜，供人们比较结果和代码。


## FrozenLake-v0

FrozenLake-v0 即冰湖是 OpenAI Gym 中经典的强化学习环境。在一个格子世界中，通过上下左右移动，躲避冰洞，到达目标。

![image](https://github.com/foamliu/Visualize-Q-Learning/raw/master/images/FrozenLake-v0.png)

## 马尔可夫决策过程(MDP)

环境通常被规范为马可夫决策过程（MDP）

环境状态的集合 S, 动作的集合 A, 初始状态 S0

在状态之间转换的规则 P(s,a,s’)
- P( [1,1], up, [1,2] ) = 0.8

规定转换后“即时奖励”的规则 r(s)
- r( [4,3] ) = +1

目标: 取得最大化的预期利益。

策略: 状态到动作的映射 S to A

## Q-学习

Q-学习是强化学习的经典算法。

Q 为效用函数（utility function），用于评价在特定状态下采取某个动作的优劣。

Q(s,a) = (1- α) Q(s,a) + α (r + γ(max(Q(s’,a’)))

其中α为学习速率（learning rate），γ为折扣因子（discount factor）。学习速率α越大，保留之前训练的效果就越少。折扣因子γ越大，所起到的作用就越大。在对状态进行更新时，会考虑到眼前利益（r），和记忆中的利益（max(Q(s’,a’))）。指记忆里下一个状态的动作中效用值的最大值。自适应动态规划(ADP)。

![image](https://github.com/foamliu/Visualize-Q-Learning/raw/master/images/Q-Learning.jpg)

## 使用方法

### 求解冰湖问题
基于 Q 学习求解冰湖问题，并把过程记录为 data.json 文件：

```bash
$ python frozenLakeQ.py
```

### 可视化求解过程
启动 Web 服务器：

```bash
$ python -m http.server
```

打开 Chrome 浏览器，地址栏输入 http://127.0.0.1:8000 ，即可看到如下所示的学习过程：

![image](https://github.com/foamliu/Visualize-Q-Learning/raw/master/images/learning_process.gif)

 

