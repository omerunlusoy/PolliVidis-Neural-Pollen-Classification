import math
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from twilio.rest import Client


# this class includes some general supporting functions for CNN
class Helper_Functions:
    def __init__(self):
        super().__init__()

    # 1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels
    def image_convert_to_numpy(self, tensor):
        image = tensor.clone().detach().cpu().numpy()  # clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
        image = image.squeeze()
        image = image.transpose(1, 2, 0)
        # print(image.shape)                                                                            # (28, 28, 1.jpg)
        # denormalize image
        image = image * np.array((0.5,)) + np.array((0.5,))
        image = image.clip(0, 1)
        return image

    def show_images(self, images, labels, classes=None, predictions=None):
        fig = plt.figure(figsize=(25, 4))

        grid = 20
        if len(images) < grid:
            grid = len(images)

        for index in np.arange(grid):
            ax = fig.add_subplot(2, math.ceil(grid/2), index + 1, xticks=[], yticks=[])
            plt.imshow(self.image_convert_to_numpy(images[index]))

            if predictions is None:
                if classes is None:
                    ax.set_title([labels[index].item()])
                else:
                    ax.set_title([classes[labels[index].item()]][0])
            else:
                ax.set_title("{} ({})".format(str(labels[index].item()), str(predictions[index].item())),
                             color=("green" if predictions[index] == labels[index] else "red"))
                # plt.savefig('dataset.jpg', dpi=500, bbox_inches='tight')
        plt.axis('off')
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

    def easter_egg(self, sample_image, Helvetica_path_=None):
        source_img = sample_image.convert("RGB")
        draw = ImageDraw.Draw(source_img)
        font = ImageFont.truetype(Helvetica_path_, 200)
        width, height = sample_image.size
        draw.text((width/5, height/2), 'Ömer loves you too!', font=font, fill='blue')
        return source_img


    def label_sample_image(self, sample_image, box_coordinates, pollens=None, plot=False, title='', no_grid=False, Helvetica_path_=None):
        source_img = sample_image.convert("RGB")
        for i, coo in enumerate(box_coordinates):
            draw = ImageDraw.Draw(source_img)
            minr, minc, maxr, maxc = coo
            draw.rectangle([(minc, minr), (maxc, maxr)], outline='blue', width=12)
            if pollens:
                # font = ImageFont.load_default()
                font = ImageFont.truetype(Helvetica_path_, 100)
                if minr - 100 > 0:
                    draw.text((minc, minr - 100), pollens[i], font=font, fill='black')
                else:
                    draw.text((minc, maxr + 30), pollens[i], font=font, fill='black')

        if plot and no_grid:
            plt.imshow(source_img)
            plt.axis('off')
            # plt.savefig('prediction.jpg', dpi=500, bbox_inches='tight')
            plt.show()

        elif plot:

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(title)

            ax.imshow(source_img)
            ax = plt.gca()
            # plt.savefig('prediction.jpg', dpi=500, bbox_inches='tight')
            # plt.axis('off')

            # Change major ticks to show every 20.
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.yaxis.set_major_locator(MultipleLocator(200))

            # Change minor ticks to show every 5. (20/4 = 5)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            # Turn grid on for both major and minor ticks and style minor slightly
            # differently.
            ax.grid(which='major', color='#CCCCCC', linestyle='--')
            ax.grid(which='minor', color='#CCCCCC', linestyle=':')

            plt.show()
        return source_img

    def send_SMS(self, text):
        account_sid = 'ACbcca60de279e3c47a6001320d2c3aafb'
        auth_token = 'd564d9b638cf11b938aba6923f2fbd08'

        twilio_number = '+12346574594'
        target_number = '+905424173804'

        client = Client(account_sid, auth_token)
        message = client.messages.create(body=text, from_=twilio_number, to=target_number)


