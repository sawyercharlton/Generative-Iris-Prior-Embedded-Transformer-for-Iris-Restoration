import torch
import time

timestamp0 = time.time()

dict = torch.load('990000.pt')["g_ema"]

timestamp1 = time.time()
print("--- %s load time 1 ---" % (timestamp1 - timestamp0))

old_val = dict["convs.11.noise.weight"]

timestamp2 = time.time()
print("--- %s search time 2 ---" % (timestamp2 - timestamp1))

print('old value: ', old_val)

#  Modify the parameter name.
dict["conv1.texture.weight"] = dict.pop("conv1.noise.weight")
dict["convs.0.texture.weight"] = dict.pop("convs.0.noise.weight")
dict["convs.1.texture.weight"] = dict.pop("convs.1.noise.weight")
dict["convs.2.texture.weight"] = dict.pop("convs.2.noise.weight")
dict["convs.3.texture.weight"] = dict.pop("convs.3.noise.weight")
dict["convs.4.texture.weight"] = dict.pop("convs.4.noise.weight")
dict["convs.5.texture.weight"] = dict.pop("convs.5.noise.weight")
dict["convs.6.texture.weight"] = dict.pop("convs.6.noise.weight")
dict["convs.7.texture.weight"] = dict.pop("convs.7.noise.weight")
dict["convs.8.texture.weight"] = dict.pop("convs.8.noise.weight")
dict["convs.9.texture.weight"] = dict.pop("convs.9.noise.weight")
dict["convs.10.texture.weight"] = dict.pop("convs.10.noise.weight")
dict["convs.11.texture.weight"] = dict.pop("convs.11.noise.weight")
dict.pop("noises.noise_0")
dict.pop("noises.noise_1")
dict.pop("noises.noise_2")
dict.pop("noises.noise_3")
dict.pop("noises.noise_4")
dict.pop("noises.noise_5")
dict.pop("noises.noise_6")
dict.pop("noises.noise_7")
dict.pop("noises.noise_8")
dict.pop("noises.noise_9")
dict.pop("noises.noise_10")
dict.pop("noises.noise_11")
dict.pop("noises.noise_12")

timestamp3 = time.time()
print("--- %s modify time ---" % (timestamp3 - timestamp2))

torch.save({"g_ema": dict}, '990000_v2.pt')

timestamp4 = time.time()
print("--- %s save time ---" % (timestamp4 - timestamp3))

#  Check if success.
changed_dict = torch.load('990000_v2.pt')["g_ema"]

timestamp5 = time.time()
print("--- %s load time 2 ---" % (timestamp5 - timestamp4))

new_val = changed_dict["convs.11.texture.weight"]
print('new value: ', new_val)
