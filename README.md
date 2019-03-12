# A deep collective Entity Linking model for entity disambiguation

### Global model tf records example 1
```
[words:]      (President Obama) and his wife (Michelle) handed out (Halloween) treats to area children and military families at the (White  House  Sunday) evening
[entities:]   (x         x    ) x   x   x    (enid_2)   x      x   (enid_3)    x      x  x    x        x   x         x       x  x   (enid_4 enid_4 enid_4) x
[mask_index:] 0
[begin_span:] [0, 5, 8, 18]
[end_span:]   [1, 5, 8, 20]
```

### Global model tf records example 2
```
[words:]      (President Obama) and   his   wife  (Michelle) handed out   (Halloween) treats to    area  children and   military families at    the   (White    House    Sunday)   evening
[entities:]   (x_x_x     x_x_x) x_x_x x_x_x x_x_x (e1_e2_e3) x_x_x  x_x_x (e4_e5_e6)  x_x_x  x_x_x x_x_x x_x_x    x_x_x x_x_x    x_x_x    x_x_x x_x_x (e7_e8_e9 e7_e8_e9 e7_e8_e9) x_x_x
[mask_index:] 0
[begin_span:] [0, 5, 8, 18]
[end_span:]   [1, 5, 8, 20]
x_x_x: string type, default entity ids
e1_e2_e3: string type, top3 ids predicted by local model
```

### Global model tf records example 3
```
[words:]      (President Obama) and his wife (Michelle) handed out   (Halloween) treats to area children and military families at the (White House Sunday) evening
[entities:]   (x         x    ) x   x   x    (e1)       x      x     (e4)        x      x  x    x        x   x        x        x  x   (e7    e7    e7)     x
[entities_1:] (x         x    ) x   x   x    (e2)       x      x     (e5)        x      x  x    x        x   x        x        x  x   (e8    e8    e8)     x
[entities_2:] (x         x    ) x   x   x    (e3)       x      x     (e6)        x      x  x    x        x   x        x        x  x   (e9    e9    e9)     x
[mask_index:] 0
[begin_span:] [0, 5, 8, 18]
[end_span:]   [1, 5, 8, 20]
entities, entities_1, entities_2: same type with example 1
```
