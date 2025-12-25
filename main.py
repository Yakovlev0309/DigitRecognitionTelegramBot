import telebot
import network as nw
import csv_convert
from PIL import Image
from config import TELEGRAM_TOKEN


model_path = nw.model_path
image_path = "image.jpg"
bot = telebot.TeleBot(TELEGRAM_TOKEN)


@bot.message_handler(commands=["start"])
def main(message):
    bot.send_message(message.chat.id, "Добро пожаловать! Отправь боту картинку (желательно, с "
                                    "разрешением 28x28), и он попробует определить, "
                                    "что за цифра на ней изображена.")


@bot.message_handler(commands=["train"])
def train(message):
    # TODO выводить прогресс обучения (например, в процентах) в одном и том же сообщении
    bot.send_message(message.chat.id, "Нейронная сеть обучается...")
    nw.train_network("data/mnist_train.csv", 60000)
    bot.send_message(message.chat.id, "Обучение закончено! Можешь отправлять "
                                    "боту изображения (желательно, с разрешением 28x28).")


@bot.message_handler(content_types=["photo"])
def run(message):
    # Получение и скачивание картинки из чата
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Открытие полученной картинки и сохранение её в виде файла на сервере
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    # Открытие сохранённой картинки, изменение её размера и сохранение полученного изображения
    img = Image.open("image.jpg")
    img = img.resize((28, 28))
    img.save(image_path)

    # Перевод изображения в формат csv
    csv_image_name = csv_convert.image_to_csv(image_path)

    # Распознавание нарисованной цифры
    result_number = recognizeDigit(csv_image_name)

    # Отправка пользователю результата в текстовом виде
    bot.send_message(message.chat.id, f"Нарисованное число: {result_number}")

    # Отправка сжатого изображения
    # bot.send_photo(message.chat.id, img)


def recognizeDigit(csv_image_name):
    return nw.run(csv_image_name, model_path)


bot.polling(non_stop=True)
