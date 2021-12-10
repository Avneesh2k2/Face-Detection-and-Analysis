import cv2


eye = r"C:\Users\Avneesh\Desktop\Project\haarcascade_eye.xml"
face = r"C:\Users\Avneesh\Desktop\Project\haarcascade_frontalface_default.xml"

face_obj = cv2.CascadeClassifier(face)
eye_obj = cv2.CascadeClassifier(eye)


img = cv2.imread(r"C:\Users\Avneesh\Desktop\Project\person4.jpg",1)


# Resize Image

# resize = cv2.resize(img,(600,600))

# Display Image

# cv2.imshow("Test",img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()

# cv2.imshow("New",resize)

# cv2.waitKey(0)

# cv2.destroyAllWindows()


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_obj.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
eyes = eye_obj.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

for x,y,w,h in faces:
    gray = cv2.rectangle(img, (x,y) , (x+w,y+h), (0,255,0),3)

for ex,ey,ew,eh in eyes:
    gray = cv2.rectangle(img, (ex,ey) , (ex+ew,ey+eh), (0,255,0),3)

print(faces)

cv2.imshow("New", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


