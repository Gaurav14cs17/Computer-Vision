import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchvision.models as models


class CNN_VGG16_Modol():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ft = models.vgg16(pretrained=True)
        print(self.model_ft)
        self.last_layer = self.model_ft.classifier[1]
        self.model_ft.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

    def get_feature(self, image):
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        #my_embedding = torch.zeros((1, 512, 7, 7))
        my_embedding = torch.zeros((1,4096))

        # 4. Define a function that will copy the output of a layer
        def copy_data(model, input, output):
            #print(output.data.cpu().numpy().shape) #1, 512, 7, 7
            my_embedding.copy_(output.data)

        # 5. Attach that function to our selected layer
        h = self.last_layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model_ft(image_tensor)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding


if __name__ == '__main__':
    model_obj = CNN_VGG16_Modol()
    from PIL import Image
    image = Image.open("DATAIMAGE/positive/crop001001a.png").convert('RGB')
    image_2 = Image.open("DATAIMAGE/positive/crop001002c.png").convert('RGB')
    image_output = model_obj.get_feature(image)
    image_output_2 = model_obj.get_feature(image_2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(image_output,image_output_2)
    print('\nCosine similarity: {0}\n'.format(cos_sim))
    print(cos_sim.shape)

