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
dicts.append(unpickle("/home/xuyekun/MAE-k/datasets/cifar-10-batches-py/data_batch_5"))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

save_dir = os.path.join('MAE-k/datasets', 'train')
os.makedirs(save_dir, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

for i in range(5):
    data = dicts[i][b'data']
    labels = dicts[i][b'labels']

    for j, img_array in enumerate(data):
        img = img_array.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        class_name = classes[labels[j]]
        img.save(os.path.join(save_dir, class_name, f"{j}.png"))