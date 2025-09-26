import cv2
import numpy as np

#load the imgs as db
image = []

img = cv2.imread('poker cards/Ace.jpg')
image.append(img)
img = cv2.imread('poker cards/Two.jpg')
image.append(img)
img = cv2.imread('poker cards/Three.jpg')
image.append(img)
img = cv2.imread('poker cards/Four.jpg')
image.append(img)
img = cv2.imread('poker cards/Five.jpg')
image.append(img)
img = cv2.imread('poker cards/Six.jpg')
image.append(img)
img = cv2.imread('poker cards/Seven.jpg')
image.append(img)
img = cv2.imread('poker cards/Eignt.jpg')
image.append(img)
img = cv2.imread('poker cards/Nine.jpg')
image.append(img)
img = cv2.imread('poker cards/Ten.jpg')
image.append(img)
img = cv2.imread('poker cards/Jack.jpg')
image.append(img)
img = cv2.imread('poker cards/Queen.jpg')
image.append(img)
img = cv2.imread('poker cards/King.jpg')
image.append(img)

#create a list of labels
classes = ['ace','two','three','four','five','six','seven','eight','nine','ten','jack','queen','king']

#create describing db
def describe(images):
    descriptor_list = []
    orb = cv2.ORB_create(nfeatures=1000)
    #extract features from each img
    for im in images:
        kp,des = orb.detectAndCompute(im,None)
        if des is not None:
            #create a list of descriptors
            descriptor_list.append(des)
        else:
            print("No features found")
    return descriptor_list

#do the match
def objclsf(frame,descriptor_list):
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(frame,None)  
    if des is None:
        print("No features found")
        return -1
    #create the matcher
    matcher = cv2.BFMatcher()
    best_matches = []
    
    #perform matching with db
    for descriptor in descriptor_list:
        matches = matcher.knnMatch(des,descriptor, k=2)              #knnmatch
        good = []
        
        for m,n in matches:
            if m.distance <n.distance* 0.75:    
                good.append([m])
    
        best_matches.append(len(good))
        
    #classId
    classID = -1
    
    if len(best_matches) > 0:
        max_val = max(best_matches)
        if max_val > 10:
            classID = best_matches.index(max_val)
        
    return classID

#functionality
descriptor_list = describe(image)
if len(descriptor_list) == 0:
    print("No features found")
    exit()
    
webcam = cv2.VideoCapture(0)

while True:
    #read the frame 
    success, frame = webcam.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    #get the class ID
    objID = objclsf(frame, descriptor_list)
    
    if objID!= -1:
        cv2.putText(frame, classes[objID], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 4)
        
    cv2.imshow('Frame',frame)
    k = cv2.waitKey(30)
    if k == ord('q'):
        break 
        
webcam.release()
cv2.destroyAllWindows()        
