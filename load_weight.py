import numpy as np
import torch



def load_weight_from_tar(path=None):
    return torch.load(path)['state_dict']

def load_weight_from_npz(path="../imagenet1k_Mixer-L_16.npz"):
    weight = np.load(path)
    weight_dict = dict()
    for i in weight:
        print(i,weight[i].shape)

        temp = torch.tensor(weight[i])
        if 'stem' in i:
            token=i.split("/")
            if 'kernel' in i:
                token[1]='weight'
                temp=temp.permute(3,2,0,1)
            name=token[0]+"."+token[1]
            weight_dict[name]=temp

        elif 'pre_head_layer_norm' in i:
            token=i.split("/")
            if('scale') in i:
                token[1]='weight'
            name=token[0]+"."+token[1]
            weight_dict[name]=temp

        elif (('head' in i) and ('pre' not in i)):
            token = i.split("/")
            if('kernel' in i):
                token[1]='weight'
                temp = temp.T
            name = token[0]+"."+token[1]
            weight_dict[name]=temp

        else:
            token=i.split("/")
            name="layer."+token[0].split("_")[1]+"."
            if('LayerNorm' in token[1]):
                name+=(token[1]+".")
                if('scale' in token[2]):
                    token[2]='weight'
                name+=token[2]
                weight_dict[name]=temp
            else:
                if('kernel' in token[3]):
                    token[3]='weight'
                name+=(token[1]+"."+token[2]+"."+token[3])
                temp = temp.T
                weight_dict[name]=temp

    return weight_dict

def load_weight_from_torch(path="/home/koc4466677/exp/Mixer_5",location='cpu'):
    k=torch.load(path,map_location=location)
    weight_dict = k.state_dict()
    return weight_dict

def SVD_split_weight(weight_dict,is_svd=False,ratio=1,height=1024,width=4096):
    New_weight_dict=weight_dict.copy()
    num_components = int(min(height,width)*ratio)
    if (is_svd):
        for i in weight_dict:
            if('channel_mixing' in i):

                name=""
                token = i.split(".")
                for j in range(len(token)-1):
                    name+=(token[j]+".")
                if('Dense_0' in i):
                    if('weight' in i):
                        S,V,D=np.linalg.svd((weight_dict[i].T).numpy())
                        SV = torch.tensor(S)@torch.diag(torch.tensor(V))

                        New_weight_dict[name+"S.weight"] = (SV[:,:num_components].T)
                        New_weight_dict[name+"D.weight"] = (torch.tensor(D[:num_components,:])).T

                    if('bias' in i):
                        New_weight_dict[name+"D.bias"] =(weight_dict[i])
                if('Dense_1' in i):
                    if('weight' in i):
                        S,V,D=np.linalg.svd((weight_dict[i].T).numpy())
                        VD = torch.diag(torch.tensor(V))@torch.tensor(D)

                        New_weight_dict[name+"S.weight"] = (torch.tensor(S[:,:num_components]).T)
                        New_weight_dict[name+"D.weight"] = (VD[:num_components,:]).T

                    if('bias' in i):
                        New_weight_dict[name+"D.bias"] =(weight_dict[i])
                if('recover' not in i):
                    New_weight_dict.pop(i)
            print(i)
    return New_weight_dict


def weight_for_SVD(New_weight_dict,SVD_Config=None,height=1024,width=4096):
    for i in (New_weight_dict.keys()):
        if ('channel_mixing' in i) and ('weight' in i):
            layer=((i.split(".")[1]))
            ratio=SVD_Config[layer]
            if (len(ratio)>1):
                ratio1,ratio2=ratio[0],ratio[1]
            else:
                ratio1 = ratio[0]
                ratio2 = ratio[0]
            num_components1 = int(min(height,width)*ratio1)
            num_components2 = int(min(height,width)*ratio2)
            if('Dense_0' in i):
                if ('.S.' in i):
                    tmp =New_weight_dict[i][:num_components1,:]
                elif ('.D.' in i):
                    tmp =New_weight_dict[i][:,:num_components1]
                #print(i,New_weight_dict[i].shape,tmp.shape)
                New_weight_dict[i]=tmp
            if('Dense_1' in i):
                if ('.S.' in i):
                    tmp =New_weight_dict[i][:num_components2,:]
                elif ('.D.' in i):
                    tmp =New_weight_dict[i][:,:num_components2]
                #print(i,New_weight_dict[i].shape,tmp.shape)
                New_weight_dict[i]=tmp
            #print(i,tmp.shape,num_components)
    return New_weight_dict

def SVD_Split_weight_ResMLP(weight_dict):
    New_weight_dict = weight_dict.copy()
    ratio = 1
    for i in weight_dict.keys():
        
            if('mlp.fc' in i ):
                current_layer = i.split('.')[1]
                #print(i,weight_dict[i].shape)
                
                
                #print(height,width)
                
                name=""
                token = i.split(".")
                for j in range(len(token)-1):
                            name+=(token[j]+".")
                if('fc1' in i):
                            if('weight' in i):
                                height,width = weight_dict[i].shape
                                num_components = int(min(height,width)*ratio)
                                S,V,D=np.linalg.svd((weight_dict[i].T).numpy())
                                SV = torch.tensor(S)@torch.diag(torch.tensor(V))

                                New_weight_dict[name+"S.weight"] = (SV[:,:num_components].T)
                                New_weight_dict[name+"D.weight"] = (torch.tensor(D[:num_components,:])).T
                                New_weight_dict.pop(i)

                            if('bias' in i):
                                New_weight_dict[name+"D.bias"] =(weight_dict[i])
                                New_weight_dict.pop(i)
                            
                if('fc2' in i):
                            if('weight' in i):
                                height,width = weight_dict[i].shape
                                num_components = int(min(height,width)*ratio)
                                S,V,D=np.linalg.svd((weight_dict[i].T).numpy())
                                VD = torch.diag(torch.tensor(V))@torch.tensor(D)

                                New_weight_dict[name+"S.weight"] = (torch.tensor(S[:,:num_components]).T)
                                New_weight_dict[name+"D.weight"] = (VD[:num_components,:]).T
                                New_weight_dict.pop(i)

                            if('bias' in i):
                                New_weight_dict[name+"D.bias"] =(weight_dict[i])
                                New_weight_dict.pop(i)
            
               
                
    return New_weight_dict

def SVD_Split_weight_ResMLP_Value(weight_dict):
    New_weight_dict = weight_dict.copy()
    ratio = 1
    value = []
    for i in weight_dict.keys():
        
            if('mlp.fc' in i ):
                current_layer = i.split('.')[1]
                #print(i,weight_dict[i].shape)
                
                
                #print(height,width)
                
                name=""
                token = i.split(".")
                for j in range(len(token)-1):
                            name+=(token[j]+".")
                if('fc1' in i):
                            if('weight' in i):
                                height,width = weight_dict[i].shape
                                num_components = int(min(height,width)*ratio)
                                S,V,D=np.linalg.svd((weight_dict[i].T).numpy())
                                value.append(V)
                                SV = torch.tensor(S)@torch.diag(torch.tensor(V))

                                New_weight_dict[name+"S.weight"] = (SV[:,:num_components].T)
                                New_weight_dict[name+"D.weight"] = (torch.tensor(D[:num_components,:])).T
                                New_weight_dict.pop(i)

                            if('bias' in i):
                                New_weight_dict[name+"D.bias"] =(weight_dict[i])
                                New_weight_dict.pop(i)
                            
                if('fc2' in i):
                            if('weight' in i):
                                height,width = weight_dict[i].shape
                                num_components = int(min(height,width)*ratio)
                                S,V,D=np.linalg.svd((weight_dict[i].T).numpy())
                                value.append(V)
                                VD = torch.diag(torch.tensor(V))@torch.tensor(D)

                                New_weight_dict[name+"S.weight"] = (torch.tensor(S[:,:num_components]).T)
                                New_weight_dict[name+"D.weight"] = (VD[:num_components,:]).T
                                New_weight_dict.pop(i)

                            if('bias' in i):
                                New_weight_dict[name+"D.bias"] =(weight_dict[i])
                                New_weight_dict.pop(i)
            
               
                
    return New_weight_dict, value

def weight_for_SVD_ResMLP(ori,SVD_Config=None):
    New_weight_dict = ori.copy()
    kc=0
    kk=0
    for i in (New_weight_dict.keys()):
        
            if('mlp.fc' in i and 'bias' not in i):
                
                current_layer =  i.split('.')[1]
                height,width = New_weight_dict[i].shape
                token = i.split(".")
                name = token[3]
                
                ratio=SVD_Config[current_layer][name]
                
                num_components = int(min(height,width)*ratio)
                
                if('fc1' in i):
                    if ('.S.' in i):
                        tmp =New_weight_dict[i][:num_components,:]
                    elif ('.D.' in i):
                        tmp =New_weight_dict[i][:,:num_components]
                    
                    New_weight_dict[i]=tmp
                    
                            
                if('fc2' in i):
                    if ('.S.' in i):
                        tmp =New_weight_dict[i][:num_components,:]
                    elif ('.D.' in i):
                        tmp =New_weight_dict[i][:,:num_components]
                    
                    New_weight_dict[i]=tmp
                    
                kc+=ori[i].flatten().shape[0]
                kk+=tmp.flatten().shape[0]
                
    print(kc,kk)
    return New_weight_dict


def num_params_for_SVD_ResMLP(ori, SVD_Config=None):
    kc = 0
    kk = 0
    for i in ori.keys():
        if 'mlp.fc' in i and 'bias' not in i:
            current_layer = i.split('.')[1]
            height, width = ori[i].shape
            token = i.split(".")
            name = token[3]

            ratio = SVD_Config[current_layer][name]

            num_components = int(min(height, width) * ratio)
            tmp = None

            if 'fc1' in i or 'fc2' in i:
                if '.S.' in i:
                    kk += ori[i].numel() * num_components // ori[i].shape[0]
                elif '.D.' in i:
                    kk += ori[i].numel() * num_components // ori[i].shape[1]

            kc += ori[i].numel()

    return (kc, kk)


def Init_Config(blocks=12,mode=1):
    SVD_Config ={}
    for i in range(blocks):
         SVD_Config[str(i)]=[mode, mode]
    return SVD_Config
def Init_Weight(path="/home/elvis08/MixerNet/Model_1K_Base_0_ratio_SVD_Full.pt"):
    New_weight_dict = load_weight_from_torch(path)
    return New_weight_dict

def Config_each_layer(mode=0.5):
    All_layer={}
    All_layer['fc1']=mode
    All_layer['fc2']=mode
    return All_layer

def Create_Model_Weight_Config(mode=1,layer=24):
    SVD_Config = {}
    
   
    for i in range(layer):
        
        SVD_Config[str(i)] = Config_each_layer(mode)
    
    return SVD_Config
