import os
import time
import cv2
import torch
import numpy as np
from Class_yolov5 import Detect
from utils.general import scale_coords



if __name__ == '__main__':
    
    weights = r'bestshow.pt'
    
    im_size = 640
    conf_thres = 0.5#0.8
    iou_thres = 0.5#0.7
    device = 'cpu'
    classes = None


    cap = cv2.VideoCapture(r'videos\check3.mp4') 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # if os.path.exists('output.avi'):
    #     os.remove('bird_output.avi')
    
    result = cv2.VideoWriter('runs\detect\output.mp4',fourcc, fps, (width,height))

    # load model
    Det = Detect(weights, im_size, device)
    points = []
    # center_list = []
    detect = False
    dis = False
    st1 = time.time()
    # ret, frame = cap.read()
    while True:
        with torch.no_grad(): 
            
            ret,frame = cap.read()   
            # frame = cv2.flip(frame, 1)
            # print('FPS: ',fps)
            if not ret:
                break
            
            # draw points on frame
            # frame = Det.draw_polygon(frame, points)
            # print(points)

            if detect:
                # ret,frame = cap.read()
                
                st = time.time()
                # center_list = []
                pred = Det.detect(frame ,conf_thres = conf_thres, iou_thres=iou_thres) # result 

                # check pred is None or not None?
                if pred != torch.tensor([]):
                    if list(pred[0]) != []: 
                        # pr = time.time()

                        # rescale boundingbox
                        pred_rescale = torch.tensor(pred[0])
                        pred_rescale = scale_coords(Det.img_size_detect[2:], pred_rescale, frame.shape).round()

                        # get center bounding box
                        centers = Det.get_center(pred_rescale)[:,:2]
                        center_draw = tuple(centers.numpy()[0])
                        centers = list(centers.numpy()[0])

                        # get center bottom bounding box
                        centerbottom = Det.get_center_bottom(pred_rescale)[:,:2]
                        centerbot_draw = tuple(centerbottom.numpy()[0])
                        centers_bot = list(centerbottom.numpy()[0])
                        
                        # initial 2 lists of people and objects
                        list_peo = []
                        list_obj = []

                        # classification people and objects into list_peo and list_obj
                        for j in range(len(pred[0])):
                            if((int(pred[0][j][5])) ==4):
                                coordinatesX = int(centerbottom[j][0])
                                coordinatesY = int(centerbottom[j][1])
                                center_peo = [coordinatesX,coordinatesY]
                                list_peo.append(center_peo)
                                cv2.circle(frame,(int(coordinatesX),int(coordinatesY)),5, (255,0,0), 5)

                                # if inside dangerous zone send notification to telegram  
                                # if Det.sInside(points,center_peo):
                                #     frame = Det.alert(frame)
                                
                            elif((int(pred[0][j][5])) ==6):
                                pass

                            else:
                                coordX = int(centerbottom[j][0])
                                coordY = int(centerbottom[j][1])
                                center_obj = [coordX,coordY]
                                list_obj.append(center_obj)
                                cv2.circle(frame,(int(coordX),int(coordY)),4, (0,0,255), 5)

                        # cv2.circle(frame,centerbot_draw,radius=5,color=(255,0,0),thickness=5)

                        # Convert original coordinates to bird coordinates
                        frame = Det.bird_detect_people(frame,list_peo,list_obj,350,width,height) 

                        
                
                print('time per frame: ',time.time()-st)
                print('FPS: ',fps)
                # print(time.time()-pr)

                # draw bounding box
                # img_rstl = Det.draw_all_box(img=frame,pred=pred)
                # img_rstl = frame
                
            else:
                img_rstl = height

            event =  cv2.waitKey(1)

            if event== ord('q'):
                break
            elif event == ord('p'):
                cv2.waitKey(-1) 
            elif event == ord('c'):
                detect = False
            elif event == ord('d'):
                # points.append(points[0])
                # points = points[:-1]
                detect = True

            # write frame to video output file
            result.write(frame)

            # show window
            cv2.imshow("Warning", frame)
            cv2.setMouseCallback('Warning', Det.handle_left_click, points)
            
    # When everything done, release the capture
    cap.release()
    result.release()
    cv2.destroyAllWindows()


