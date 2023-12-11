from io import BytesIO
import requests
import telebot
# import network1 as nw
import network2 as nw
import csv_convert
from PIL import Image

model_path = "model.npz"
image_path = "image.jpg"
token = ""
bot = telebot.TeleBot(token)


@bot.message_handler(commands=["start"])
def main(message):
    # TODO проверка на наличие модели (если модели нет - обучить) (обучение модели должно быть отдельной функцией)
    bot.send_message(message.chat.id, "Добро пожаловать! Нейронная сеть обучается...")
    nw.train_network("data/mnist_train.csv", 60000)
    # TODO выводить прогресс обучения (например, в процентах)
    bot.send_message(message.chat.id, "Сеть обучена! Отправь боту картинку с "
                                      "разрешением 28x28, и он попробует определить, "
                                      "что за число на ней изображено.")


@bot.message_handler(content_types=["photo"])
def image(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # TODO приводить полученную картинку к размеру 28 x 28

    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    img = Image.open("image.jpg")
    img = img.resize((28, 28))
    img.save(image_path)

    csv_image_name = csv_convert.image_to_csv(image_path)
    # csv_image_name = "7.csv"
    # csv_image_name = "4.csv"

    result_number = nw.run(csv_image_name, model_path)
    bot.send_message(message.chat.id, f"Нарисованное число: {result_number}")

    # img.save("new_image.jpg")

    # f = open("new_image.jpg")
    bot.send_photo(message.chat.id, img)


bot.polling(non_stop=True)
