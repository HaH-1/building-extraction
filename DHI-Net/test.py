import BodyEdge as be


unloader = be.transforms.ToPILImage()

transform = be.transforms.Compose([
    be.transforms.ToTensor(),

])

test_path = './data/whu/test'
model_path = './mdl/DHI/0.mdl'

test_dataset = be.whu_dataset_val(test_path,transform=transform)
test_dataloader = be.DataLoader(test_dataset,batch_size=1)

my_model = be.Model()


my_model.load_state_dict(be.torch.load(model_path))

device= 'cuda' if be.torch.cuda.is_available() else 'cpu'
my_model.to(device)

for i, (image, label) in enumerate(test_dataloader):
    # 训练数据的个数
    count_image = len(test_dataloader.dataset)
    image, label = image.to(device), label.to(device)
    my_model.eval()
    output = my_model(image)
    output = be.nn.Sigmoid()(output[0])

    image1 = unloader(output.round().cpu().clone().squeeze(0))
    image1.save('./data/result2/{}__.png'.format(i))