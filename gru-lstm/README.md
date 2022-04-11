1. GRU-scratch
2. GRU-concise
3. LSTM-scratch
4. LSTM-concise

<br/>

- GRU-scratch:  
还是比较简单的，就是那几个 R, Z, H_tilta, O 的计算，还是需要记得 Weight_Clip

- GRU-concise:  
还是需要记得 forward 函数中不需要遍历 num_step， 不过同样需要展平，
方便后面的 Linear层。 H 隐藏元初始化时候是三维的，多一个层数变量

- LSTM-scratch:  
主要是 C、H初始化的时候多了个隐变量

- LSTM-concise:  
forward 时候直接传入 state就行， state=(C,H)

- 双向循环RNN 和 Deep-RNN
```python
if not self.rnn.bidirectional:
    self.num_directions = 1
    self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
else:
    self.num_directions = 2
    self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens)
```
- 

