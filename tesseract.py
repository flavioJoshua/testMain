import cv2
import pytesseract
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


#se nn lo metti va in errore 
# export TESSDATA_PREFIX=/home/flavio/.conda/envs/yolo3.10/share/tessdata


def get_text(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    testo = pytesseract.image_to_string(img, lang='ita')
    return testo

# Avvia la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('e'):
        break
## Chiude la webcam
cap.release()
cv2.destroyAllWindows()

with ThreadPoolExecutor() as executor:
    future = executor.submit(get_text, frame)
    print(future.result())
