# A deep collective Entity Linking model for entity disambiguation

### Global model tf records example
```
[words:]      (President Obama) and his wife (Michelle) handed out (Halloween) treats to area children and military families at the (White  House  Sunday) evening
[entities:]   (x         x    ) x   x   x    (enid_2)   x      x   (enid_3)    x      x  x    x        x   x         x       x  x   (enid_4 enid_4 enid_4) x
[mask_index:] 0
[begin_span:] [0, 5, 8, 18]
[end_span:]   [1, 5, 8, 20]
```