import torch
import crypten

crypten.init()

x = torch.tensor([1.0, 2.0, 3.0])
x_enc = crypten.cryptensor(x) # encrypt

x_dec = x_enc.get_plain_text() # decrypt

y_enc = crypten.cryptensor([2.0, 3.0, 4.0])
sum_xy = x_enc + y_enc # add encrypted tensors
sum_xy_dec = sum_xy.get_plain_text() # decrypt sum

print("Success!")
