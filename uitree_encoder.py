import json
from sentence_transformers import SentenceTransformer
import torch
import os
import numpy as np
import tqdm
from torch.utils.data import Dataset

from dataset.rico_utils import get_all_labeled_uis_from_rico_screen, ScreenInfo
from dataset.rico_dao import load_rico_screen_dict
from UI_embedding.UI2Vec import HiddenLabelPredictorModel
from torch.utils.data.sampler import SubsetRandomSampler
from autoencoder import ScreenLayoutDataset, LayoutAutoEncoder, LayoutTrainer
from UI_embedding.plotter import plot_loss


class UITreeFeatures():
    def __init__(self, ui_text, ui_class):
        self.ui_text = ui_text
        self.ui_class = ui_class

    def to_numpy(self):
        return np.full()


class UITreeDataset(Dataset):
    def __init__(self, ascreens):
        self.screens = ascreens

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, index):   # TODO ui_text is currently dropped
        return self.screens[index].ui_class #torch.from_numpy(self.screens[index].to_numpy()).type(torch.FloatTensor)


### config
screen = "C:\projects\Screen2Vec\Rico/filtered_traces/com.instagram.android/trace_1/view_hierarchies/1017.json"
ui_model = "UI2Vec_model.ep120"
screen_model = "Screen2Vec_model_v4.ep120"
layout_model = "layout_encoder.ep800"
num_predictors = 4
net_version = 4

args = {  # default data from README
  #  "dataset":"C:\projects\Screen2Vec\Rico/filtered_traces/com.instagram.android/trace_1/view_hierarchies",
    "dataset":"C:\projects\Screen2Vec\Rico/some_ui_trees",
    "batch_size":256,
    "epochs":400,
    "rate":0.001,
    "type":0
}


### preprocessing json -> numeric data

## takes path to a screen json,
## and returns 3 arrays (one element coresponds to one box in the UI): encoded text, box type, bounding box edge coords
def get_screen_components_vector(screen):
    # convert screen jsons to list of boxes
    with open(screen) as f:
        rico_screen = load_rico_screen_dict(json.load(f))
    labeled_text = get_all_labeled_uis_from_rico_screen(rico_screen)
    # => labeled_text: list of boxes [ ... [text content, component type, array of bounding box edge coords] ... ]
  #  bert = SentenceTransformer('bert-base-nli-mean-tokens') # TODO uncomment this to include text in encoding
  #  bert_size = 768

   # loaded_ui_model = HiddenLabelPredictorModel(bert, bert_size, 16)
   # loaded_ui_model.load_state_dict(torch.load(ui_model, map_location=torch.device('cpu')), strict=False)

    ui_class = torch.tensor([UI[1] for UI in labeled_text])
    ui_text = [UI[0] for UI in labeled_text]
    ui_coords = torch.tensor([UI[2] for UI in labeled_text])

    # TODO my quick text encoder: take the length
    ui_text = [len(atext) for atext in ui_text]
   # UI_embeddings = loaded_ui_model.model([ui_text, ui_class])  # TODO change this to encoder -> outside of this method

    return [ui_text, ui_class, ui_coords]


# single screen execution:
res = get_screen_components_vector("C:\projects\Screen2Vec\Rico/filtered_traces/com.instagram.android/trace_1/view_hierarchies/1017.json")

# multi screen execution:
dataset_path = "C:\projects\Screen2Vec\Rico/some_ui_trees"
screens = []
for fn in os.listdir(dataset_path):
    if fn.endswith('.json'):
        screen_data = get_screen_components_vector(dataset_path + '/' + fn)#  ScreenLayout(dataset_path + '/' + fn)
        screens.append(UITreeFeatures(screen_data[0], screen_data[1]))
        # UITreeFeatures represents one (steady) screen. However TODO the box coords are currently discarded



### autoencoder tech
# screens / a UITreeDataset represents a sequence of steady screens
dataset = UITreeDataset(screens) #ScreenLayoutDataset(args['dataset'])

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=test_sampler)

model = LayoutAutoEncoder()

trainer = LayoutTrainer(model, train_loader, test_loader, args['rate'])

train_loss_data = []
test_loss_data = []
for epoch in tqdm.tqdm(range(args['epochs'])):
    print("--------")
    print(str(epoch) + " loss:")
    train_loss = trainer.train(epoch)
    print(train_loss)
    print("--------")
    train_loss_data.append(train_loss)

    test_loss = trainer.test(epoch)
    test_loss_data.append(test_loss)
    print(test_loss)
    print("--------")

    if (epoch % 50) == 0:
        print("saved on epoch " + str(epoch))
        trainer.save(epoch)
plot_loss(train_loss_data, test_loss_data, "output/autoencoder")
trainer.save(args['epochs'], "output/autoencoder")