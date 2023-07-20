import cv2                        #imported the cv2 file (import name for open cv used for image analysis)
import numpy as np                
import webbrowser
import math
import time
import docx
import time

doc = docx.Document()
handhaar = cv2.CascadeClassifier("filepath")
recording = cv2.VideoCapture(0)           #Capturing the video 
while (recording.isOpened()):
    #while camera is open for recording we will be reading the image
    start_time =time.time()
    ret, img = recording.read() 
    img = cv2.flip(img, 1)

    # From the Video we will be getting the hand data in the rectangular box on the window.
    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    cropped_image = img[100:300, 100:300]

    # Transform the image into grayscale image
    Grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # After transforming into grayscale now we will be blurring the image using gaussian blur.
    value = (35, 35)                    #kernel size of gaussian filter h=35 w =35
    blurred = cv2.GaussianBlur(Grayscale_image, value, 0)

    # Otsu Thresholding and binary thresholding Method is used to threshold the image.
    _, threshed_img = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # show thresholded image
    cv2.imshow('Thresholded_image', threshed_img)

    # To avoid error in unpacking , we are checking OpenCV version 
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(threshed_img.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if version == '4':
        contours, hierarchy = cv2.findContours(threshed_img.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Now we will be finding the contour with the maximum area 
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour 
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

    #Now we will be finding convex hull of the hand region
    hull = cv2.convexHull(max_contour)
    areahull = cv2.contourArea(hull)
    area_max_contour = cv2.contourArea(max_contour)

    arearatio = ((areahull - area_max_contour) / area_max_contour) * 100

    #After finding the hand region we now will draw contours(max contour as green colour and hull colour as red)
    draw_contour = np.zeros(cropped_image.shape, np.uint8)
    cv2.drawContours(draw_contour, [max_contour], -1, (0, 255, 0), 1)
    cv2.drawContours(draw_contour, [hull], -1, (0, 0, 255), 1)

    #Again will find the convex hull to get the indices of the contour points 
    hull = cv2.convexHull(max_contour, returnPoints=False)

    #Finding convexity defects of the maximum contour and convex hull of the hand 
    defects = cv2.convexityDefects(max_contour, hull)
    count_defects = 0
    cv2.drawContours(threshed_img, contours, -1, (0, 255, 0), 3)

    # Using the Cosine Rule to determine the angle for every defect (the space between fingers).
    # Disregard defects and focus only on angles greater than 90 degrees.
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        # apply cosine rule here
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(cropped_image, far, 1, [0, 0, 255], -1)
        # dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(cropped_image, start, end, [0, 255, 0], 2)
        # cv2.circle(crop_img,far,5,[0,0,255],-1)

    # define actions required can't keep count defect 0 because it will print 0 &1 both on the camera opening
    # if count_defects == 0:
    #    cv2.putText(img, "1", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    if count_defects == 1:
        cv2.putText(img, "2", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        face_only = handhaar.detectMultiScale(img, 1.15, 5)
        print(face_only)
        if len(face_only) == 1:
            cv2.imwrite('captured.png', img)
        for x, y, w, h in face_only:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    elif count_defects == 2:
        cv2.putText(img, "3", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say your youtube search query")
            audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            print("You said: " + query)
            webbrowser.open('https://www.youtube.com/results?search_query=' + query, new=2)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Speech Recognition service; {0}".format(e))
 
    elif count_defects == 3:
        cv2.putText(img, "4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        import speech_recognition as sr
        import pyaudio

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('speak something: ')
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print('you said:{}'.format(text))
            #add text to word document 
            doc.add_paragraph(text)
            #save the word document 
            doc.save('my_doc.docx')
            print("text saved to my_doc.docx")
        except:
            print('sorry could not recognize your voice')     #External dependencies: If your code relies on external libraries , network latency or issues with those dependencies can cause your program to slow down or fail.

    elif count_defects == 4:
        cv2.putText(img, "5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        
        import speech_recognition as sr
       
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say your google search query")
            audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            print("You said: " + query)
            webbrowser.open('https://www.google.com/search?q=' + query, new=2)
            print("Google search opened in browser. Waiting for user input...")
        except sr.UnknownValueError:  
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Speech Recognition service; {0}".format(e))
    else:
        if area_max_contour < 2000:
            cv2.putText(img, 'Put hand in the box', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(img, '0', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                
                cv2.putText(img, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                webbrowser.open('https://www.google.com/calendar/')
    frame_time = time.time() - start_time
    print(f" Gesture Processing time: {frame_time:.5f} seconds")
    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((draw_contour, cropped_image))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(2)
    if k == 27:
        break





































































                # import time
                # import speech_recognition as sr

                # timer_running = False
                # start_time = 0

                # # initialize speech recognizer
                # r = sr.Recognizer()

                # # function to get voice input
                # def get_voice_input():
                #     with sr.Microphone() as source:
                #         print("Say how many minutes of timer you want to set : ")
                #         audio = r.listen(source)
                #         try:
                #             voice_input = r.recognize_google(audio)
                #             return voice_input
                #         except sr.UnknownValueError:
                #             print("Sorry, I did not understand that.")
                #             return None
                #         except sr.RequestError as e:
                #             print("Sorry, could not request results from Google Speech Recognition service; {0}".format(e))
                #             return None

                # # main loop
                # while True:
                #     # get user input for timer duration
                #     if not timer_running:
                #         voice_input = get_voice_input()
                #         if voice_input:
                #             try:
                #                 timer_duration = int(voice_input)
                #                 start_time = time.time()
                #                 timer_running = True
                #                 print("Timer set for {} minutes".format(timer_duration))
                #             except ValueError:
                #                 print("Sorry, I did not understand that. Please say a valid number of minutes.")
    
                #     # check if timer has expired
                #     if timer_running and (time.time() - start_time) >= timer_duration * 60:
                #         print("Timer expired!")
                #         timer_running = False
                #         start_time = 0
                #         # do something when timer expires, e.g., display a message on screen or play a sound