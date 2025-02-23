import clip
import torch
import numpy as np
from collections import OrderedDict
import json
import pandas as pd


def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id


def text_prompt(dataset='HMDB51', clipbackbone='ViT-B/16', device='cpu'):
    actionlist, actionprompt, actiontoken = [], {}, []
    numC = {'HMDB51-feature-30fps-center': 51, 'HVU': 66}

    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False

    # convert to token, will automatically padded to 77 with zeros
    if dataset == 'HMDB51-feature-30fps-center':
        meta = open("../data/HMDB51/HMDB51_action.list", 'rb')
        actionlist = meta.readlines()
        meta.close()
        actionlist = np.array([a.decode('utf-8').split('\n')[0] for a in actionlist])
        actiontoken = np.array([convert_to_token(a) for a in actionlist])
    elif dataset == 'HVU':
        label_df = pd.read_csv("/ssd4/ajaved/evaluation_repos/open_vocab/label_files/open_vocab_all_categories_labels.csv")  # Assuming the label file is in this location
        actionlist = label_df['label'].tolist()  # Extract action labels
        numC['HVU'] = len(actionlist)  # Set dynamic class count for HVU
        actiontoken = np.array([convert_to_token(a) for a in actionlist])  # Tokenize action labels
    # More datasets to be continued

    # query the vector from dictionary
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(numC[dataset]))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(numC[dataset]))

    return actionlist, actiondict, actiontoken

