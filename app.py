# https://core.telegram.org/bots#
# https://pytorch.org/get-started/locally/
# pip3 install opencv-python
# pip3 install PyTelegramBotAPI==2.2.3


import telebot
import traceback
import torch
from torchvision import transforms
import config
from handler import *
import cv2

bot = telebot.TeleBot(config.TOKEN)
classes=['human', 'no_human']
model = torch.jit.load('humans_video_25ep.pt')
transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_photo(message):
    photo = message.photo[1].file_id
    file_info = bot.get_file(photo)
    file_content = bot.download_file(file_info.file_path)
    return file_content

def get_video(message):
    video = message.document.file_id
    file_info = bot.get_file(video)
    file_content = bot.download_file(file_info.file_path)
    src = "upload/123" + message.document.file_name;
    with open(src, 'wb') as new_file:
        new_file.write(video)
    bot.send_message(message.chat.id, "Файл загружен и готов к обработке")
    return file_content

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Здравствуйте! Данная нейросеть определяет наличие человека на фото. \nАвтор: @jaremy_gk')

@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_content = get_photo(message)
        image = byte2image(file_content)
        image = transform(image)
        print(image)
        model.eval()
        image = torch.unsqueeze(image, 0)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        if format(classes[int(preds)]) == 'human':
            bot.send_message(message.chat.id,text='Человек на территории') # : {}'.format(classes[int(preds)]))
        else:
            bot.send_message(message.chat.id, text='Ничего не обнаружено')
    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Упс, что-то пошло не так :( Обратитесь в службу поддержки!')

@bot.message_handler(content_types=['document'])
def repeat_all_messages1(message):
    try:
        file_content = get_video(message)
        vid_capture = cv2.VideoCapture(file_content)
        while (vid_capture.isOpened()):
            ret, frame = vid_capture.read()
            if ret == True:
                t = frame
                image = byte2image(t)
                image = transform(image)
                model.eval()
                image = torch.unsqueeze(image, 0)
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                if format(classes[int(preds)]) == 'human':
                    bot.send_message(message.chat.id, text='Человек на территории') # {}'.format(classes[int(preds)]))
                else:
                    bot.send_message(message.chat.id, text='Ничего не обнаружено')

    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Упс, что-то пошло не так :( Обратитесь в службу поддержки!')





if __name__ == '__main__':
    import time
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            time.sleep(15)
            print('Restart!')
