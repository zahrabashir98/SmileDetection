import sys
import cv2
from keras.models import load_model
from matplotlib import pyplot as plt


model = load_model("models/model.h5")


def find_faces(image):
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    face_rects = face_cascade.detectMultiScale(
        image,
        scaleFactor = 1.1,
        minNeighbors = 22
    )
    return face_rects


def load_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image, gray_image


def predict(gray_image):
    face_rects = find_faces(gray_image)

    for face_rect in face_rects:
        x, y, w, h = face_rect
        face = gray_image[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48)).reshape((1, 48, 48, 1))
        predicted_emotions = model.predict(face)[0]
        best_emotion = 'smiling' if predicted_emotions[1] > predicted_emotions[0] else 'non-smiling'

        # Create a json serializable result
        yield dict(
            border = dict(
                x = float(x),
                y = float(y),
                width = float(w),
                height = float(h),
            ),
            prediction = {'smiling': float(predicted_emotions[0]), 'non-smiling': float(predicted_emotions[1])},
            emotion = best_emotion
        )


def put_text(image, rect, text):
    x, y, w, h = rect

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 30.0
    font_thickness = int(round(font_scale * 1.5))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    center_text_x = x + (w // 2)
    center_text_y = y + (h // 2)
    text_w, text_h = text_size

    lower_left_text_x = center_text_x - (text_w // 2)
    lower_left_text_y = center_text_y + (text_h // 2)

    cv2.putText(
        image, text,
        (lower_left_text_x, lower_left_text_y),
        font, font_scale, (0, 255, 0), font_thickness
    )


def draw_face_info(image, face_info):
    x = int(face_info['border']['x'])
    y = int(face_info['border']['y'])
    w = int(face_info['border']['width'])
    h = int(face_info['border']['height'])
    emotion = face_info['emotion']

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    put_text(image, (x, y, w, h // 5), emotion)


def show_image(image, title='Result'):
    plt.subplot(111), plt.imshow(image), plt.title(title)
    plt.show()



if __name__ == '__main__':
    image, gray_image = load_image(sys.argv[1])
    for face_info in predict(gray_image):
        print(face_info)
        draw_face_info(image, face_info)
    show_image(image)
