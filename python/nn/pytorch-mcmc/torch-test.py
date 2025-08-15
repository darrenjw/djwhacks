#!/usr/bin/env python3

import torch
x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())

if torch.cuda.is_available():
    cx = x.to("cuda")
    print(x)
    print(cx)


