import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn
import torchvision.models as models
import pickle
class MultiOutputModel(nn.Module):
    def __init__(self, num_colors, num_types, num_seasons, num_genders):
        super(MultiOutputModel, self).__init__()

        
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])  # till conv5_x

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
        )

        
        self.color_head = nn.Linear(512, num_colors)
        self.type_head = nn.Linear(512, num_types)
        self.season_head = nn.Linear(512, num_seasons)
        self.gender_head = nn.Linear(512, num_genders)

    def forward(self, x):
        x = self.backbone(x)    
        x = self.pool(x)        
        x = self.flatten(x)     
        x = self.shared_fc(x)   

        return {
            'baseColour': self.color_head(x),
            'articleType': self.type_head(x),
            'season': self.season_head(x),
            
            'gender': self.gender_head(x),
        }

# Load label encoders
with open('label_encoders.pkl', 'rb') as f: 
    label_encoders = pickle.load(f)

idx2season =label_encoders['season'].classes_
idx2gender = label_encoders["gender"].classes_
idx2article = label_encoders["articleType"].classes_
idx2color = label_encoders["baseColour"].classes_

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiOutputModel(num_seasons=len(idx2season), num_genders=len(idx2gender), num_types=len(idx2article), num_colors=len(idx2color))
model.load_state_dict(torch.load("output_models/best_model.pth", map_location=device)) # give path of the saved model

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),             # resize for validation (no randomness)
    transforms.ToTensor(),
    
])
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred = {
            "season": idx2season[torch.argmax(outputs['season'], dim=1).item()],
            "gender": idx2gender[torch.argmax(outputs['gender'], dim=1).item()],
            "articleType": idx2article[torch.argmax(outputs['articleType'], dim=1).item()],
            "baseColor": idx2color[torch.argmax(outputs['baseColour'], dim=1).item()],
        }
    return pred

if __name__ == "__main__":
    uvicorn.run(app,port=8000, host="localhost")