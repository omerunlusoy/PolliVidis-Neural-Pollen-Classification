from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import timedelta, datetime
import cv2  # for video
from torchvision.utils import save_image


# this class includes some general supporting functions for CNN
class Helper_Functions:
    def __init__(self):
        super().__init__()

    # 1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels
    def image_convert_to_numpy(self, tensor):
        image = tensor.clone().detach().cpu().numpy()  # clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
        image = image.squeeze()
        image = image.transpose(1, 2, 0)
        # print(image.shape)                                                                            # (28, 28, 1)
        # denormalize image
        image = image * np.array((0.5,)) + np.array((0.5,))
        image = image.clip(0, 1)
        return image

    def show_images(self, images, labels, classes=None, predictions=None):
        fig = plt.figure(figsize=(25, 4))

        for index in np.arange(20):
            ax = fig.add_subplot(2, 10, index + 1, xticks=[], yticks=[])
            plt.imshow(self.image_convert_to_numpy(images[index]))

            if predictions is None:
                if classes is None:
                    ax.set_title([labels[index].item()])
                    # plt.savefig('trainset.jpg', dpi=500, bbox_inches='tight')
                else:
                    ax.set_title([classes[labels[index].item()]][0])
                    # plt.savefig('trainset.jpg', dpi=500, bbox_inches='tight')
            else:
                ax.set_title("{} ({})".format(str(labels[index].item()), str(predictions[index].item())),
                             color=("green" if predictions[index] == labels[index] else "red"))
                # plt.savefig('dataset.jpg', dpi=500, bbox_inches='tight')
        plt.show()

    def plot_loss_and_corrects_epoch(self, epochs, losses, corrects, validation_losses, validation_corrects):
        plt.title('Epoch vs Loss')
        plt.plot(range(epochs), losses, label="training loss")
        plt.plot(range(epochs), validation_losses, label="validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Loss.jpg', dpi=500, bbox_inches='tight')
        plt.show()

        plt.title('Epoch vs Corrects')
        plt.plot(range(epochs), corrects, label="training accuracy")
        plt.plot(range(epochs), validation_corrects, label="validation accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Correct')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Corrects.jpg', dpi=500, bbox_inches='tight')
        plt.show()
