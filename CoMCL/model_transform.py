
import torch

#对于moco或其他非resnet变体

def simCLRToPretrainModel(model,save_path):
    for name,p in list(model.items()):
        temp = name.split('.')
        head = temp[0]
        temp = temp[1:]
        new_name = '.'.join(temp)
        if head == 'head':
            model.pop(name)
        else:
            if 'fc' in new_name:
                print("remove fc")
                model.pop(name)
            else:
                model[new_name] = model.pop(name)

        torch.save(model,save_path)

def mocoToPretrainModel(model,save_path):
    for name, p in list(model.items()):
        #print(name)
        temp1 = name.split('.')
        head = temp1[0]    #记录backbone/head
        temp = temp1[2:]
        new_name = '.'.join(temp)
        if head == 'encoder_k':
            model.pop(name)
        else:
            if 'fc' in new_name:
                print("remove fc")
                model.pop(name)
            else:
                model[new_name] = model.pop(name)

    torch.save(model,save_path)


#model = torch.load('../Lesion-based-CL/checkpoints_student/best_validation_model.pt')
model = torch.load('./try2/best_validation_model.pt')
#model = checkpoint['model']
save_path = '../pytorch_classification-ffversion/pretrain_model/moco_newtry2_best.pt'

#simCLRToPretrainModel(model,save_path)
mocoToPretrainModel(model,save_path)

