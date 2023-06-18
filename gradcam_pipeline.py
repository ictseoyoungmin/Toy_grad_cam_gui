import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
from torchvision.models import resnet50
from gradcam import GradCam 

class GradCAMPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model_test()
        self.transform = self._transform()
        
    def load_model_test(self):
        model = resnet50(pretrained=True)
        model.to(self.device)
        model.eval()
        return model
    
    def load_model(self,model_path):
        model = torch.load(model_path,map_location='cpu')
        model.to(self.device)
        model.eval()
        self.model = model
        
    def _transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform
    
    def process_image(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path,mode='RGB')
        else:
            raise ValueError("Unsupported image type. Expecting file path (str) or NumPy array.")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def visualize_gradcam_test(self,image_path, gradcam):
        import matplotlib.pyplot as plt
        
        if isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = np.array(Image.open(image_path))
            image = cv2.resize(image,(224,224))
            
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(gradcam, cmap='jet', alpha=0.7)
        plt.imshow(image, alpha=0.5)
        plt.title('Grad-CAM')
        plt.axis('off')
        
        # plt.show()
        plt.savefig('test.jpg')

        
    def run(self, image_path):
        image_tensor = self.process_image(image_path)
        
        # class 매핑
        # 여기서 class 매핑을 수행하는 코드를 작성
        
        # Grad-CAM 결과 생성
        gradcam = GradCam(self.model,-1)
        gradcam_result = gradcam.generate_gradcam(image_tensor)
        
        # 결과 출력
        # 여기서 Grad-CAM 결과를 출력하는 코드를 작성
        
        # self.visualize_gradcam_test(image_path, gradcam_result)
        return gradcam_result
        
if __name__ == "__main__":
    import cv2
    pipeline = GradCAMPipeline()
    image_path = r"C:\Users\201721360\PythonWork\pytorch_serving\example.jpg"  # 이미지 경로를 적절히 지정
    image = cv2.resize(cv2.cvtColor(cv2.imread(filename=image_path),cv2.COLOR_BGR2RGB),(224,224))

    pipeline.run_pipeline(image)
