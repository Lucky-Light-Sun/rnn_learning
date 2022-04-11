# rnn_learning
This is the rnn learning note

1. the dataset
2. the basic rnn
3. the gru (*gated recurent unit*)
4. the lstm (*long short term memory*)
5. bi
6. deep rnn

下面我们主要分块进行代码书写

- sequence.py:介绍  
`sin(0.01x) + norm` 的时序预测

- text-preprocessing.py:  
对文本预处理，包括字符转换、词元划分tokenize、字典的建立、下标到词元的双向映射

- language-models-and-dataset:  
验证除了一元语法词，单词序列也遵循齐普夫定律，并建立起两个data_iter的函数，并进行了细致的封装

- rnn:  
RNN是对隐变量进行循环计算的网络
常常使用困惑度来衡量语言模型的质量
使用了 grid_clip，但是注意和 torch.cuda.amp 的冲突性 

- rnn-scratch:
注意这个 .T 转置
predict 需要进行预热处理，并且设置 batch_size=1
有趣的是，我们仅仅是在数据集获取dataloader时候，强调了 num_step 
`tau`，其实在训练过程中，完全不关tau的大小，可以视为0，这也是我们预测过程中做的事情
当使用顺序划分时，我们需要分离梯度以减少计算量。
离谱的是 BCE 需要 Softmax，BCEWithLogitsLoss 不需要 Softmax， CrossEntropyLoss 也不需要Softmax

- rnn-concise:
主要是使用了 nn.rnn, 然后在该层的基础上，加入输出层次
一定呀注意input和output的形状，对应我们损失函数的情况