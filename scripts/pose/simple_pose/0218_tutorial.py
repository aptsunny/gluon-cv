from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils

# from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose


print(1581341591.6628928-1581341591.662819)
print(1581341591.6628928-1581341591.6626186)

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

# pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)
pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])

# x = mx.nd.ones((1, 3, 256, 192))


im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/pose/soccer.png?raw=true',
                          path='soccer.png')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)


class_IDs, scores, bounding_boxs = detector(x)

# print(detector.summary(x))

# pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
# predicted_heatmap = pose_net(pose_input)
# pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
# ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
#                               class_IDs, bounding_boxs, scores,
#                               box_thresh=0.5, keypoint_thresh=0.2)
# plt.show()

pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)
predicted_heatmap = pose_net(pose_input)

print(pose_net.summary(pose_input))

pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
plt.show()

