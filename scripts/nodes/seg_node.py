#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
GROUNDED_SAM_2_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'submodules', 'Grounded-SAM-2'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
sys.path.append(GROUNDED_SAM_2_PATH)
import cv2
import rospy
import sys
import argparse
import json
import numpy as np
import supervision as sv
import torch
from cv_bridge import CvBridge
from torchvision.ops import box_convert

from utils import PROJECT_NAME
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from scripts.nodes import \
    ImageSeg, ImageSegRequest, ImageSegResponse

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import grounding_dino.groundingdino.datasets.transforms as T
from grounding_dino.groundingdino.util.inference import load_model, predict

def grounded_sam2_callback(req: ImageSegRequest) -> ImageSegResponse:

    bridge = CvBridge()

    cv_image = None
    if config['dataset']['format'] == 'strcam':
        # Structured Camera
        cv_image = cv2.imread(req.cur_data_path +
                              f"/out_rgb.png", cv2.IMREAD_UNCHANGED)
    else:
        # Simulation RGB-D camera
        origin_image = rospy.wait_for_message(config['dataset']['color_topic'], Image)
        cv_image = bridge.imgmsg_to_cv2(
            origin_image, desired_encoding="passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    image = np.array(cv_image, dtype=np.uint8)

    # Predict classes and hyper-param for GroundingDINO
    rospy.loginfo(f"target is {req.object_name}")
    if req.object_name[-1] != '.':
        req.object_name += '.' # make sure the object name ends with a period
        
    # preporcess image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = PILImage.fromarray(image)
    image_transformed, _ = transform(image_source, None)

    sam2_predictor.set_image(image)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_transformed,
        caption=req.object_name,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    
    if boxes is None or boxes.numel() == 0:
        rospy.logwarn(f"Not find any {req.object_name} in the image.")
        masks_response = ImageSegResponse()
        # masks_response.masks_image
        masks_response.result = False
        return masks_response
    
    # process the box prompt for SAM 2
    h, w, _ = image.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    
    """
    Visualize image with supervision useful API
    """
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=cv_image.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(GROUNDED_SAM_2_PATH, "/groundingdino_annotated_image.png"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections, )
    cv2.imwrite(os.path.join(GROUNDED_SAM_2_PATH, "/grounded_sam2_annotated_image_with_mask.png"), annotated_frame)

    n, height, width = detections.mask.shape
    uint8_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(n):
        # TODO: Just save the first mask
        if i == 0:
            uint8_mask[detections.mask[i] == True] = i + 1
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[detections.mask[i] == True] = 255
            cv2.imwrite(os.path.join(req.cur_data_path + f'/mask.png'), mask)
    masks_image = uint8_mask.astype(np.uint8)
    
    bridge2 = CvBridge()
    masks_response = ImageSegResponse()
    masks_response.masks_image = bridge2.cv2_to_imgmsg(masks_image, encoding="mono8")
    masks_response.result = True
    rospy.loginfo(f'Segmentation Success! Image includes {n} objects.')
    return masks_response

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} seg node.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--gpu_id',
                        type=int,
                        required=True,
                        help='Specify gpu id.')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='Debug mode, save evaluation results.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        rospy.logwarn('No GPU available.')
        device = torch.device('cpu')
        
    with open(args.config) as f:
        config = json.load(f)
        
    # build SAM2 image predictor
    sam2_checkpoint = config['grounded_sam']['sam2_checkpoint']
    sam2_model_cfg = config['grounded_sam']['sam2_model_cfg']
    print(f'Loading SAM2 model from {sam2_checkpoint}')
    print(f'Loading SAM2 model config from {sam2_model_cfg}')
    sam2_model = build_sam2(sam2_model_cfg, os.path.join(GROUNDED_SAM_2_PATH, sam2_checkpoint), device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # build grounding dino model
    grounding_dino_cfg = config['grounded_sam']['grounding_dino_cfg']
    grounding_dino_checkpoint = config['grounded_sam']['grounding_dino_checkpoint']
    print(f'Loading GroundingDINO model from {grounding_dino_checkpoint}')
    print(f'Loading GroundingDINO model config from {grounding_dino_cfg}')
    box_threshold = config['grounded_sam']['box_threshold']
    text_threshold = config['grounded_sam']['text_threshold']
    grounding_model = load_model(
        model_config_path=os.path.join(GROUNDED_SAM_2_PATH, grounding_dino_cfg), 
        model_checkpoint_path=os.path.join(GROUNDED_SAM_2_PATH, grounding_dino_checkpoint),
        device=device
    )
    
    rospy.loginfo("Load Grounded-SAM2: Done!")
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.DEBUG if bool(args.debug) else rospy.INFO)
    grounded_sam2_service = rospy.Service('grounded_sam2', ImageSeg, grounded_sam2_callback)
    rospy.loginfo(f'{PROJECT_NAME} Grounded-SAM2 service started.')
    rospy.spin()
    rospy.loginfo(f'{PROJECT_NAME} seg node finished.')