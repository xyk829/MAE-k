import os
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dicts = []
print(unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_1").keys())
dicts.append(unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_1"))
dicts.append(unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_2"))
dicts.append(unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_3"))
dicts.append(unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_4"))
val_dict = unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_5")
test_dict = unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/test_batch")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

save_dir = os.path.join('MAE-k/datasets', 'train')
os.makedirs(save_dir, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

for i in range(4):
    data = dicts[i][b'data']
    labels = dicts[i][b'labels']

    for j, img_array in enumerate(data):
        img = img_array.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        class_name = classes[labels[j]]
        img.save(os.path.join(save_dir, class_name, f"{j}.png"))

val_dir = os.path.join('MAE-k/datasets', 'val')
os.makedirs(val_dir, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
 
val_data = val_dict[b'data']
val_labels = val_dict[b'labels']
for i, img_array in enumerate(val_data):
    img = img_array.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    class_name = classes[val_labels[i]]
    img.save(os.path.join(val_dir, class_name, f"{i}.png"))
    
test_dir = os.path.join('MAE-k/datasets', 'test')
os.makedirs(test_dir, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
 
test_data = test_dict[b'data']
test_labels = test_dict[b'labels']
for i, img_array in enumerate(test_data):
    img = img_array.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    class_name = classes[test_labels[i]]
    img.save(os.path.join(test_dir, class_name, f"{i}.png"))