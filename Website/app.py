import os
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model(
    'vgg_best.keras')

app.config['UPLOAD_FOLDER'] = 'static/images/'


@app.route('/')
def index():
    titles, images, paragraphs, dates = get_news()
    news = []
    for i in range(3):
        news.append({
            'title': titles[i].text.strip(),
            'link': "https://www.news-medical.net/"+titles[i].a['href'],
            'image': images[i]['src'],
            'paragraph': paragraphs[i].text.strip(),
            'date': dates[i].text.strip()
        })
    return render_template('index.html', news=news)


@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    uploaded_image = None
    result = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                uploaded_image = filename
                image = cv2.imread(
                    os.path.join(app.config['UPLOAD_FOLDER'], filename))
                task_type = request.form.get('task', 'classification')

                if task_type == 'classification':
                    new_image = cv2.resize(image, (64, 64))
                    new_image = np.expand_dims(new_image, axis=0)

                    result = model.predict(new_image)
                    result = np.argmax(result, axis=1)[0]
                    if result == 1:
                        result = "This image is abnormal."
                    elif result == 0:
                        result = "This image is normal."
                    return render_template('diagnosis.html', uploaded_image=uploaded_image, result=result)
                elif task_type == 'detection':
                    image_copy = image.copy()
                    window_size = (64, 64)
                    stride = 64
                    height, width, _ = image.shape
                    predictions = []
                    for y in range(0, height - window_size[1] + 1, stride):
                        for x in range(0, width - window_size[0] + 1, stride):
                            image_patch = image[y:y +
                                                window_size[1], x:x+window_size[0]]
                            image_patch = cv2.resize(image_patch, (64, 64))
                            image_patch = np.expand_dims(image_patch, axis=0)
                            prediction = model.predict(image_patch)
                            prediction = np.argmax(prediction, axis=1)
                            predictions.append(prediction)
                            if prediction == 1:
                                cv2.rectangle(image_copy, (x, y),
                                              (x+window_size[1], y+window_size[1]), (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(
                        'static/results/', filename), image_copy)

                    count = 0
                    for i in predictions:
                        if i == 1:
                            count += 1
                    total = len(predictions)
                    return render_template('diagnosis.html', uploaded_image=uploaded_image, result_image=filename, count=count, total=total)

                # chuẩn hóa ảnh
                # image = image / 255.0
                # image_copy = image.copy()
                # window_size = (256, 256)
                # stride = 256
                # height, width, _ = image.shape
                # predictions = []

                # for y in range(0, height - window_size[1] + 1, stride):
                #     for x in range(0, width - window_size[0] + 1, stride):
                #         image_patch = image[y:y +
                #                             window_size[1], x:x+window_size[0]]
                #         image_patch = cv2.resize(image_patch, (64, 64))
                #         image_patch = np.expand_dims(image_patch, axis=0)
                #         prediction = model.predict(image_patch)
                #         prediction = np.argmax(prediction, axis=1)
                #         predictions.append(prediction)
                #         if prediction == 1:
                #             cv2.rectangle(image_copy, (x, y),
                #                           (x+window_size[1], y+window_size[1]), (255, 0, 0), 2)

                # cv2.imwrite(os.path.join(
                #     'static/results/', filename), image_copy)

                # # cắt ảnh ra thành các ảnh con có kích thước 64x64
                # cropped_images = []
                # window_size = 64
                # stride = 64
                # # chuẩn hóa ảnh
                # image = image / 255.0
                # for y in range(0, image.shape[0] - window_size + 1, stride):
                #     for x in range(0, image.shape[1] - window_size + 1, stride):
                #         cropped_image = image[y:y+window_size, x:x+window_size]
                #         cropped_images.append(cropped_image)
                # cropped_images = np.array(cropped_images)
                # result = model.predict(cropped_images)
                # result = np.argmax(result, axis=1)

                # if 1 in predictions:
                #     result = "Abnormal"
                # else:
                #     result = "Normal"

                # # đếm số 1 trong predictions
                # count = 0
                # for i in predictions:
                #     if i == 1:
                #         count += 1
                # total = len(predictions)

                # new_image = cv2.resize(image, (64, 64))
                # new_image = np.expand_dims(new_image, axis=0)

                # result = model.predict(new_image)
                # result = np.argmax(result, axis=1)[0]
                # if result == 1:
                #     result = "This image is abnormal."
                # elif result == 0:
                #     result = "This image is normal."
    else:
        return render_template('diagnosis.html')


@app.route('/symtoms')
def symtoms():
    return render_template('symtoms.html')


@app.route('/treatment')
def treatment():
    return render_template('treatment.html')


@app.route('/prevention')
def prevention():
    return render_template('prevention.html')


@app.route('/definition')
def definition():
    return render_template('definition.html')


def get_news():
    # URL của trang web cần lấy nội dung
    url = "https://www.news-medical.net/?tag=/Gastric-Cancer"

    # Gửi yêu cầu GET đến trang web
    response = requests.get(url)

    # Kiểm tra mã trạng thái của yêu cầu
    if response.status_code == 200:
        # Phân tích cú pháp HTML của trang web
        soup = BeautifulSoup(response.content, "html.parser")

    divs = soup.find_all(
        'div', {'class': 'posts publishables-list-wrap first-item-larger'})

    for div in divs:
        dates = div.find_all('span', {'class': 'article-meta-date'})
        # for date in dates:
        #     print('Ngày tháng:', date.text.strip())

        paragraphs = div.find_all('p', {'class': 'hidden-xs item-desc'})
        # for paragraph in paragraphs:
        #     print('Đoạn văn:', paragraph.text.strip())

        titles = div.find_all('h3')
        # for title in titles:
        #     print('Tiêu đề:', title.text.strip())
        #     print('Link:', title.a['href'])

        images = div.find_all('img')
        # for image in images:
        #     print('Link ảnh:', image['src'])
    return titles, images, paragraphs, dates


if __name__ == '__main__':

    app.run(debug=True)
