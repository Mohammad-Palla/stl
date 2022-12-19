import cv2

# capture frames from a video
# cap = cv2.VideoCapture("Car-Detection-Basic-Open-CV/carv.mp4")


# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier("./Car-Detection-Basic-Open-CV/carx.xml")

# loop runs if capturing has been initialized.
# while True:
#     ret, frames = cap.read()
#     gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
#     cars = car_cascade.detectMultiScale(gray, 1.1, 1)
#     for (x, y, w, h) in cars:
#         cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     cv2.imshow('sKSama', frames )
#     if cv2.waitKey(33) == 27:
#         break

img = cv2.imread("Car-Detection-Basic-Open-CV/cars_img3.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.002, 3)
count = 0
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    count = count + 1
cv2.imshow('Lane 1', img )
print(f'Lane 1 => {count} vehicles')

# 2nd lane 
img2 = cv2.imread("Car-Detection-Basic-Open-CV/cars_img2.jpeg")
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.002, 3)
count2 = 0
for (x, y, w, h) in cars:
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2)
    count2 = count2 + 1
cv2.imshow('Lane 2', img2 )
print(f'Lane 1 => {count2} vehicles')

img3 = cv2.imread("Car-Detection-Basic-Open-CV/cars.png")
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.002, 3)
count3 = 0
for (x, y, w, h) in cars:
    cv2.rectangle(img3, (x, y), (x+w, y+h), (0, 0, 255), 2)
    count3 = count3 + 1
cv2.imshow('Lane 3', img3 )
print(f'Lane 1 => {count3} vehicles')


total = count + count2 + count3
print(f'Lane 1 => {round((count * 180)/total)} sec')
print(f'Lane 2 => {round((count2 * 180)/total)} sec')
print(f'Lane 3 => {round((count3 * 180)/total)} sec')
cv2.waitKey(0)
# De-allocate any associated memory usage
cv2.destroyAllWindows()
