import cv2
import PIL
import torch
import imutils
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from scipy import ndimage
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision import transforms
import matplotlib.patches as patches

#############################
#        Parameters         #
#############################
STD_R = 0.229
STD_G = 0.224
STD_B = 0.225
MEAN_R = 0.485
MEAN_G = 0.456
MEAN_B = 0.406

MASK_MAPPING_RGB = {
    (0, 0, 0): 0,  # background
    (128, 128, 0): 1,  # field
    (0, 128, 0): 2,  # lines
}
INVERSE_MASK_MAPPING_RGB = {
    0: (0, 0, 0),  # background
    1: (128, 128, 128),  # field
    2: (255, 255, 255),  # lines 
}
MASK_MAPPING_LABEL = {
    (0,0,0): 0,  # background
    (1,1,1): 1,  # field
    (2,2,2): 2,  # lines
    (3,3,3): 1,  # ball as a field
}
CHANNEL_TO_LABEL_DETECTION = {
    0:'Ball', 
    1:'Robot',
    2:'Goal Post' 
}
LABEL_TO_CHANNEL_DETECTION = {
    "ball": 0, 
    "robot": 1, 
    "goalpost": 2
}

# Error tolerance for each class
# (Ball, Robot, Goalpost)
PIXEL_TOLERANCE = [10,10,10]

#############################
#      Pre-processing       #
#############################
def resize_image(image,h=480,w=640):
    """
        Resize images to new resolution
    """
    t = transforms.Resize((h,w), interpolation=transforms.functional.InterpolationMode.NEAREST)
    #t = transforms.Resize((h,w), interpolation=PIL.Image.NEAREST)
    return np.array(t(image))


def resize_det(images,targets,h=480,w=640):
    t = transforms.Resize((h,w), interpolation=transforms.functional.InterpolationMode.NEAREST)
    
    original_size = min(images[2], images[3])
    scale_factor = original_size / transform.size
    
    for idx, box in enumerate(targets['boxes']):
            box = (box / scale_factor).long()
            targets['boxes'][idx] = box
    
    return t(images),targets

def read_image(path: str) -> np.array:
    """
        Reads in an image as a numpy array
    """
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def default_transforms() -> transforms.Compose:
    """
        Returns the default transformations that should be
        applied to any image passed to the model
    """
    return transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize(mean=[MEAN_R, MEAN_G, MEAN_B],
                                                    std=[STD_R, STD_G, STD_B])])

def reverse_normalize(image: torch.tensor) -> torch.tensor:
    """
        Reverses the normalization applied on an image.
    """
    reverse = transforms.Normalize(mean=[-MEAN_R / STD_R, -MEAN_G / STD_G, -MEAN_B / STD_B],
                                   std=[1 / STD_R, 1 / STD_G, 1 / STD_B])
    return reverse(image)


def convert_image_to_label(image) -> np.array:
    """
        Converts the segmentation image to its pixel 
        labeled equivalent using MASK_MAPPING dict
    """
    
    out = (np.zeros(image.shape[:2]))
    if (128 in np.unique(image)):
        for k in MASK_MAPPING_RGB:
            out[(image == k).all(axis=2)] = MASK_MAPPING_RGB[k]
    else:
        for k in MASK_MAPPING_LABEL:
            out[(image == k).all(axis=2)] = MASK_MAPPING_LABEL[k]

    return out


def convert_label_to_image(label) -> np.array:
    """
        Converts the pixel labeled image back to its 
        RGB equivalent using the MASK_MAPPING dict
    """
    out = (np.zeros((label.shape[0], label.shape[1], 3)))

    for k in INVERSE_MASK_MAPPING_RGB:
        out[(label == k)] = INVERSE_MASK_MAPPING_RGB[k]

    return out


def xml_to_csv(xml_folder: str, output_file: str = 'labels.csv'):
    """
        Converts a folder of XML label files into a CSV file. 
        Each XML file should correspond to an image and contain 
        the image name, image size, image_id and the names and
        bounding boxes of the objects in the image, if any.
        Any other data is in the XML files is ignored.
    """
    xml_list = []
        
    for image_id, xml_file in enumerate(glob(xml_folder, recursive=True)):
        tree = ET.parse(xml_file)
        root = tree.getroot()
                
        # Change root based on xml format
        if (root.tag == 'annotations'):
            root = root.find('image')
        
        filename = root.find('filename').text
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Annotated objects tag
        objects = root.findall('object')
        
        if not objects:
            xml_list.append((xml_file[:-4]+".jpg", -1, -1, -1, -1, -1, -1, -1, image_id))
        else:
            for member in root.findall('object'):
                box = member.find('bndbox')
                label = member.find('name').text

                xml_list.append((xml_file[:-4]+".jpg", 
                                 width, 
                                 height, 
                                 label, 
                                 int(box.find('xmin').text),
                                 int(box.find('ymin').text), 
                                 int(box.find('xmax').text),
                                 int(box.find('ymax').text), 
                                 image_id))
    
    print("Processed: ", image_id, " images")

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names).to_csv(output_file, index=None)

    
#############################
#         Training          #
#############################
def downsample(tensor):
    """
        Downsample tensor to 0.25
    """
    return torch.nn.functional.interpolate(tensor,
                                           scale_factor=0.25,
                                           mode="nearest",
                                           recompute_scale_factor=False)


def total_variation(img, weight, channels:list):
    """
        Total variation loss
    """
    total_sum = 0
    bs_img, _, h_img, w_img = img.size()    
    
    for channel in channels:
        total_sum += torch.pow(img[:,channel,1:,:]-img[:,channel,:-1,:], 2).sum()
        total_sum += torch.pow(img[:,channel,:,1:]-img[:,channel,:,:-1], 2).sum()
        
    return weight * total_sum/(bs_img*len(channels)*h_img*w_img)


def mse_loss(prediction, target):
    """
        MSE loss
    """
    return ((prediction - target)**2).sum()/prediction.shape[0]


def to_device(data, device):
    """
        Move tensor(s) to chosen device
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


#############################
#       Visualization       #
#############################
def show_labeled_image(image: torch.tensor or np.array, boxes: torch.tensor, labels: list = None):
    """
        Show the image along with the specified boxes around detected objects.
        Also displays each box's label if a list of labels is provided.
    """
    fig, ax = plt.subplots(1)

    # If the image is already a tensor, convert it 
    # back to a PILImage and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)

    ax.imshow(image)

    # Show a single box or multiple if provided
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    # Convert labels to list
    if labels is not None and not (isinstance(labels, list) or isinstance(labels, tuple)):
        labels = [labels]

    # Plot each box
    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    plt.show()
    
    
def show_image_and_probmap(image: torch.tensor or np.array, probmap: torch.tensor):
    """
        Show the image along with its predictions.
    """
    fig, ax = plt.subplots(1, 2)

    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = image.numpy().transpose((1, 2, 0))
        
    ax[0].imshow(image)
    ax[1].imshow(probmap.numpy().transpose((1, 2, 0)))

    plt.show()
    
def show_image_target_and_probmap(image:torch.tensor or np.array, target:torch.tensor, predicted:torch.tensor):
    """
        Show the image along with its targets and predictions.
    """
    fig, ax = plt.subplots(1, 3)
    fig.tight_layout()

    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = image.numpy().transpose((1, 2, 0))
        
    ax[0].imshow(image)
    ax[1].imshow(target.numpy().transpose((1, 2, 0)))
    ax[2].imshow(predicted.numpy().transpose((1, 2, 0)))

    plt.show()

    
def show_image_and_seg(image: torch.tensor or np.array, labels: np.array):
    """
        Show the image along with its segmentation.
    """
    fig, ax = plt.subplots(1, 2)

    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = image.numpy().transpose((1, 2, 0))
        
    labels = labels.numpy().transpose((1, 2, 0))


    ax[0].imshow(image)
    ax[1].imshow(convert_label_to_image(labels[:,:,0]).astype(np.uint8))

    plt.show()
    
def show_image_target_and_seg(image: torch.tensor or np.array, target:torch.tensor, predicted:torch.tensor):
    """
        Show the image along with its segmentation.
    """
    fig, ax = plt.subplots(1, 3)

    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = image.numpy().transpose((1, 2, 0))
        
    target = target.numpy().transpose((1, 2, 0))
    predicted = predicted.numpy().transpose((1, 2, 0))
        
    ax[0].imshow(image)
    ax[1].imshow(convert_label_to_image(target[:,:,0]).astype(np.uint8))
    ax[2].imshow(convert_label_to_image(predicted[:,:,0]).astype(np.uint8))

    plt.show()
  

#############################
#      Post-processing      #
#############################
def squared_distance(p1, p2):
    """
        Squared distance between points
    """
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def merge_centroids(points, d=20):
    """
        Fuse nearby centroids
    """
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if squared_distance(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((int(point[0]), int(point[1])))
    return ret


def get_predicted_centers(img):
    """
        Compute the local-maxima of the probability map
    """
    points = []
    ret,thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        cnt = contours[i]
        
        M = cv2.moments(cnt)
        
        if M['m00'] == 0: 
            continue
            
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        points.append((centroid_y,centroid_x))
    
    return merge_centroids(points)

def within_tolerance(target, prediction, tolerance):
    """
        Check if target and prediction are
        within pixel tolerance
    """
    return abs(target[0]-prediction[0]) <= tolerance and abs(target[1]-prediction[1]) <= tolerance

    
#############################
#    Evaluation Metric      #
#############################
def detection_metric(model, loader):
    """
        Detection metric
    """    
    # Stats for each channel
    fp = [0,0,0]
    tp = [0,0,0]
    fn = [0,0,0]
    tn = [0,0,0]
    
    for image, target in loader:
        # Target and output
        downsampled_target = downsample(target)
        _,output = model(image.cuda())
        
        pred_annotations = {'goalpost':[],'ball':[],'robot':[]}
        
        for label, channel in LABEL_TO_CHANNEL_DETECTION.items():
            old_fn = fn[channel]
            old_fp = fp[channel]
            
            # Target centroids
            target_channel = downsampled_target.numpy().copy()[0, channel] * 255
            target_channel = target_channel.astype(np.uint8)
            true_centroids = get_predicted_centers(target_channel)
            
            # Predicted centroids
            predicted_channel = output[0,channel].cpu()
            predicted_channel = predicted_channel.detach().numpy()
            predicted_channel = np.where(predicted_channel < 0.5, 0, predicted_channel) * 255
            predicted_channel = predicted_channel.astype(np.uint8)
            pred_centroids = get_predicted_centers(predicted_channel)
            
            pred_annotations[label] = pred_centroids
            
            # calculate TN
            if len(pred_centroids) == len(true_centroids) and len(pred_centroids) == 0:
                tn[channel] += 1
            
            # calculate TP,FP,FN
            total = len(pred_centroids)
            correct = 0
            for t_ct in true_centroids:
                found = False
                for p_ct in pred_centroids: 
                    if within_tolerance(t_ct, p_ct, PIXEL_TOLERANCE[channel]):
                        found = True
                        tp[channel] += 1
                        correct += 1
                        break
               
                if not found:
                    fn[channel] += 1
            
            fp[channel] += total - correct
            
    total_f1 = 0
    total_accuracy = 0
    total_recall = 0
    total_precision = 0
    total_fdr = 0
    
    for channel,label in CHANNEL_TO_LABEL_DETECTION.items():
    
        if tp[channel] + fn[channel] == 0:
            recall = 0  
        else:
            recall = tp[channel]/(tp[channel] + fn[channel])  
        total_recall += recall
        print(f'{label}    \tRecall:{recall:.3f}')
        
        if tp[channel] + fp[channel] == 0:
            precision = 0
        else:
            precision = tp[channel]/(tp[channel] + fp[channel]) 
        total_precision += precision
        print(f'\t\tPrecision:{precision:.3f}')
        
        if (tp[channel] + fp[channel] + tn[channel] + fn[channel]) ==0:
            accuracy = 0 
        else:
            accuracy = (tp[channel] + tn[channel])/(tp[channel] + fp[channel] + tn[channel] + fn[channel])
        total_accuracy += accuracy
        print(f'\t\tAccuracy: {accuracy:.3f}')
        
        if (precision+recall) ==0:
            f1=0
        else:
            f1=2*precision*recall/(precision+recall)
        total_f1 += f1
        print(f'\t\tF1 Score:{f1:.3f}') 
        
        fdr = 1-precision
        total_fdr += fdr
        print(f'\t\tFDR:{fdr:.3f}')
        
    total_f1 /= 3
    total_accuracy /= 3
    total_recall /= 3
    total_precision /= 3
    total_fdr /= 3
    
    return total_f1,total_accuracy,total_recall,total_precision,total_fdr

def segmentation_metric(model, dataloader):
    """
        Segmentation metric
    """
    
    total = correct = 0
    correct_field = total_field = 0
    correct_lines = total_lines = 0
    correct_background = total_background = 0
    tp_field = pred_field = fp_field = fn_field = 0
    tp_lines= pred_lines = fp_lines = fn_lines = 0
    tp_background = pred_background = fp_background = fn_background = 0

    for batch_idx,batch in enumerate(dataloader):
        # Unpack batch
        images, targets = batch
        images = images.cuda()

        # Downsample targets using nearest neighbour method
        downsampled_target = torch.nn.functional.interpolate(targets,
                                                             scale_factor=0.25,
                                                             mode="nearest",
                                                             recompute_scale_factor=False)
        downsampled_target = torch.squeeze(downsampled_target,dim=1) #(batch_size,1,H,W) -> (batch_size,H,W)
        downsampled_target = downsampled_target.type(torch.LongTensor) # convert the target from float to int

        # Run forward pass
        output, _ = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(output, 1)

        
        predicted = predicted.cpu()

        # Total correct predictions
        correct_predictions = (predicted == downsampled_target)
        correct += correct_predictions.sum().item()
        field_mask = downsampled_target.cpu()==1
        correct_field += (np.logical_and(correct_predictions,field_mask)).sum().item()
        lines_mask = downsampled_target.cpu()==2
        correct_lines +=(np.logical_and(correct_predictions,lines_mask)).sum().item()
        background_mask = downsampled_target.cpu()==0
        correct_background +=(np.logical_and(correct_predictions,background_mask)).sum().item()

        # Total number of labels
        total += (downsampled_target.size(0)*downsampled_target.size(1)*downsampled_target.size(2))
        total_field += field_mask.sum().item()
        total_lines += lines_mask.sum().item()
        total_background += background_mask.sum().item()

        # Calculate TP, FP, FN
        tp_field += (np.logical_and(correct_predictions,field_mask)).sum().item()
        pred_field = predicted.cpu() == 1
        fp_field += (np.logical_and(pred_field ,np.logical_not(field_mask))).sum().item()
        fn_field += (np.logical_and(np.logical_not(np.logical_and(correct_predictions,field_mask)),field_mask)).sum().item()

        tp_lines += (np.logical_and(correct_predictions,lines_mask)).sum().item()
        pred_lines = predicted.cpu() == 2
        fp_lines += (np.logical_and(pred_lines ,np.logical_not(lines_mask))).sum().item()
        fn_lines += (np.logical_and(np.logical_not(np.logical_and(correct_predictions,lines_mask)),lines_mask)).sum().item()

        tp_background += (np.logical_and(correct_predictions,background_mask)).sum().item()
        pred_background = predicted.cpu() == 0
        fp_background += (np.logical_and(pred_background ,np.logical_not(background_mask))).sum().item()
        fn_background += (np.logical_and(np.logical_not(np.logical_and(correct_predictions,background_mask)),background_mask)).sum().item()

    accuracy = {}    
    accuracy['Field'] = 100 * (correct_field / total_field)
    accuracy['Lines'] = 100 * (correct_lines / total_lines)
    accuracy['Background'] = 100 * (correct_background / total_background)
    accuracy['Total'] = 100 * (correct / total)
    iou = {}
    iou['Field'] = tp_field / (tp_field+fp_field+fn_field)
    iou['Lines'] = tp_lines / (tp_lines+fp_lines+fn_lines)
    iou['Background'] = tp_background / (tp_background+fp_background+fn_background)
    iou['Total'] = (iou['Field'] + iou['Lines'] +iou['Background'])/3
    
    print("Accuracy",accuracy)
    print("IOU",iou)
    
    return accuracy, iou
