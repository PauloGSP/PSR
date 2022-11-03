import argparse
import cv2
import numpy as np
import copy
import json
import time
from colorama import Fore, Style
import random


def processImage(ranges, image):
    # receives an image converts to a binary in order to find the center of the largest area
    #range
    mins = np.array([ranges['B']['min'], ranges['G']['min'], ranges['R']['min']])
    maxs = np.array([ranges['B']['max'], ranges['G']['max'], ranges['R']['max']])

    # masking
    mask = cv2.inRange(image, mins, maxs)
    mask = mask.astype(bool)

    # applies mask
    proc_img = copy.deepcopy(image)
    proc_img[np.logical_not(mask)] = 0

    # get binary image with threshold the values not in the mask
    _, proc_img = cv2.threshold(proc_img, 1, 255, cv2.THRESH_BINARY)

    return proc_img, mask


def numericPainter(painter, w, h, last_color):
    # create numeric paint and the evaluation value
    eval_painter = painter.copy()
    num_lines = random.randint(2,2)
    points = []
    colors = [(255,0,0), (0,255,0), (0,0,255)]

    for i in range(num_lines+1):

        
        if i != num_lines:
            points.append([(random.randint(int(w/num_lines)*i,int(w/num_lines)*(i+1)), 0), (random.randint(int(w/num_lines)*i,int(w/num_lines)*(i+1)), w)])
        else:
            points.append([[w,0], [w,h]])

        # define start and end point for the polygon
        if i == 0:
            sp = [0,0]
            ep = [0,h]
        else:
            sp = points[i-1][0]
            ep = points[i-1][1]

        # gets all points
        pts = np.array([sp, points[i][0], points[i][1], ep], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # color randomizer for each gap
        color = colors[random.randint(0,2)]
        while last_color == color:
            color = colors[random.randint(0,2)]

        eval_painter = cv2.fillPoly(eval_painter, [pts], color)
        point_txt = (int((points[i][1][0] - sp[0])/2))+sp[0], int((ep[1] - sp[1])/2)

        # painter
        painter = cv2.putText(painter, str(colors.index(color)+1), point_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        # save last color for next polygon
        last_color = color

    # create lines
    for i in range(len(points)):
        cv2.line(eval_painter, points[i][0], points[i][1], (0,0,0), 3, -1)
        cv2.line(painter, points[i][0], points[i][1], (0,0,0), 3, -1)

    temp = painter.copy()

    # number of pixels for the evaluation
    total_pixels = np.sum(np.equal(eval_painter, painter).astype(np.uint8))

    # print different colors and their number
    print("Colors:\n"+   Fore.BLUE +"1 - Blue\n"+   Fore.GREEN +"2 - Green\n"+   Fore.RED +"3 - Red" + Style.RESET_ALL)
    return painter, eval_painter, temp, total_pixels

def commands(canvas):
    #simple function that creates a screen with the commands 
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 1
    
    commands = ["q -> quit", "+ -> increase brush size", "- -> decrease brush size", "r -> change color to red",
                "g -> change color to green", "b -> change color to blue", "c -> clear canvas", 
                "w -> write image in file", "e -> erase", "s -> switch brush on/off"]

    for c in range(len(commands)):
        org = (50, 40*(c+1))
        canvas = cv2.putText(canvas, commands[c], org, font, fontScale, color, thickness, cv2.LINE_AA)

    return canvas

def main():

    # parser
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', required=True, dest='JSON', help='Full path to json file.')
    parser.add_argument('-v', '--video-canvas', required=False, help='Use video streaming as canvas', action="store_true", default=False)
    parser.add_argument('-p', '--paint-numeric', required=False, help='Use a numerical canvas to paint', action="store_true", default=False)
    parser.add_argument('-s', '--use-shake-detection', required=False, help='Use shake detection', action="store_true", default=False)
    parser.add_argument('-m', '--use-mouse', required=False, help='Use mouse as brush instead of centroid', action="store_true", default=False)

    args = vars(parser.parse_args())
    print(args)

    print('-> ' + Fore.RED + 'PSR ' + Fore.GREEN + 'Augmented Reality Painter' + Style.RESET_ALL)
    count = 0
    capture = cv2.VideoCapture(0)
    window_name = 'Original'
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
    window_name_paint = 'Painter'
    cv2.namedWindow(window_name_paint,cv2.WINDOW_AUTOSIZE)
    window_name_segmented = 'Segmented'
    cv2.namedWindow(window_name_segmented,cv2.WINDOW_AUTOSIZE)
    window_name_area = 'Largest Area'
    cv2.namedWindow(window_name_area,cv2.WINDOW_AUTOSIZE)
    window_name_commands = 'Commands'
    cv2.namedWindow(window_name_commands,cv2.WINDOW_AUTOSIZE)

    f = open(args["JSON"])
    data = json.load(f)

    _, frame = capture.read()

    size_brush = 5
    last_point = None
    brush = True

    height, width = frame.shape[0:2]
    color = (0,0,0)

    painter = np.ones((height, width, 3), np.uint8) * 255
    
    threshold_shake_detection = 1600

    # mask
    mr = np.zeros((height, width, 3))

    # mouse coordinates for mouse brush
    mouse_cords = {'x': None, 'y': None}

    if args['use_mouse']:
        def mouseHoverCallback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                param['x'] = width - int(x)
                param['y'] = int(y)
        cv2.setMouseCallback(window_name_paint, mouseHoverCallback, mouse_cords)

    if args["paint_numeric"]:
        painter, eval_painter, temp, total_pixels = numericPainter(painter, width, height, color)


    while True:

        ret, image = capture.read()
        cam_output = image
        k = cv2.waitKey(1)

        if not ret:
            print(Fore.RED + "failed to grab frame" + Style.RESET_ALL)
            break

        image_p, mask = processImage(data["limits"], image)

        centroid = None

        # use mouse as brush
        if args['use_mouse'] and mouse_cords['x'] is not None:
            centroid = (mouse_cords['x'], mouse_cords['y'])

        con = 4  
        # Perform the operation
        # Find the largest non background component.0 is the background label.
        
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY),  con, cv2.CV_32S)
        if nb_components > 1:
            count = 0
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
            centroid = (int(centroids[max_label][0]), int(centroids[max_label][1]))

            mr = np.equal(labels, max_label)
            b,g,r = cv2.split(image)
            b[mr] = 0
            r[mr] = 0
            g[mr] = 200
            cam_output = cv2.merge((b,g,r))

            cv2.line(cam_output, (centroid[0]+5, centroid[1]), (centroid[0]-5, centroid[1]), (0,0,255), 5, -1)
            cv2.line(cam_output, (centroid[0], centroid[1]+5), (centroid[0], centroid[1]-5), (0,0,255), 5, -1)
        else:
            mr = np.zeros((frame.shape[0], frame.shape[1], 3))
            if count == 0:
                print( Fore.RED + "Please place your object in front of the camera!" + Style.RESET_ALL)
                count += 1

        if last_point is not None and centroid is not None:
            # calculate squared dist
            distance = (last_point[0]-centroid[0])**2 + (last_point[1]-centroid[1])**2

            # oscilation detection
            if distance > threshold_shake_detection and args['use_shake_detection']:
                cv2.circle(painter, centroid, size_brush, color, -1)
            else:
                if brush:
                    cv2.line(painter, last_point, centroid, color, size_brush, -1)
        last_point = centroid


        if args["video_canvas"]:
            mask = np.not_equal(cv2.cvtColor(painter, cv2.COLOR_BGR2GRAY), 255)
            mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
            output = image.copy()
            output[mask] = painter[mask]
        else:
            output = painter

        cam_output = cv2.flip(cam_output, 1)  
        output = cv2.flip(output, 1)  

        cv2.imshow(window_name, cam_output)
        cv2.imshow(window_name_paint, output)
        cv2.imshow(window_name_commands, commands(np.ones((frame.shape[0], frame.shape[1], 3)) * 0))
        cv2.imshow(window_name_segmented, image_p)
        cv2.imshow(window_name_area, mr.astype(np.uint8)*255)

        #really wish there was a switch :c
        if k == ord("q"): #quit
            print("Key Selected: "+Fore.YELLOW+"q"+Fore.RED+"\n\tEnding program")
            break
        elif k == ord("+"): #big brush
            print("Key Selected: "+Fore.YELLOW+"+"+"\n\tIncreasing"+Style.RESET_ALL+" brush size")
            size_brush += 1
        elif k == ord("-"):#small brush
            print("Key Selected: "+ Fore.YELLOW+"-"+"\n\tDecreasing"+Style.RESET_ALL+" brush size")
            size_brush = max(2, size_brush-1)
        elif k == ord("r"):#benfica
            print("Key Selected: "+ Fore.YELLOW+"r"+Style.RESET_ALL+"\n\tChanging color to "+ Fore.RED+"RED"+Style.RESET_ALL)
            color = (0,0,255)
        elif k == ord("g"):#sporting
            print("Key Selected: "+ Fore.YELLOW+"g"+Style.RESET_ALL+"\n\tChanging color to "+ Fore.GREEN+"GREEN"+Style.RESET_ALL)
            color = (0,255,0)
        elif k == ord("b"):#porto
            print("Key Selected: "+ Fore.YELLOW+"b"+Style.RESET_ALL+"\n\tChanging color to "+ Fore.BLUE+"BLUE"+Style.RESET_ALL)
            color = (255,0,0)
        elif k == ord("c"):#clear
            print("Key Selected: "+ Fore.YELLOW+"c"+Style.RESET_ALL+"\n\tClearing canvas")
            painter = np.ones((height, width, 3), np.uint8) * 255
            if args["paint_numeric"]:
                painter = temp
        elif k == ord("w"):#save
            file_name = f"drawing_{(time.ctime(time.time())).replace(' ', '_')}.png"
            print("Key Selected: "+ Fore.YELLOW+"w"+Style.RESET_ALL+"\n\tWriting to file " +   Fore.GREEN + file_name + Style.RESET_ALL)
            cv2.imwrite(file_name, output)
            if args["paint_numeric"]:
                max_pixels = (frame.shape[0] * frame.shape[1] * 3) - total_pixels
                total_pixels = np.sum(np.equal(eval_painter, painter).astype(np.uint8)) - total_pixels

                accuracy = ((total_pixels / max_pixels) * 100)

                print("Accuracy: "+ Fore.GREEN+ str(round(accuracy,2))+Style.RESET_ALL+"%")

                if accuracy < 40:
                    print("Evaluation: "+   Fore.RED +"Not Sattisfactory - D" + Style.RESET_ALL)
                elif accuracy < 60:
                    print("Evaluation: " +   Fore.CYAN +"Satisfactory - C" + Style.RESET_ALL)
                elif accuracy < 80:
                    print("Evaluation: " +   Fore.BLUE +"Good - B" + Style.RESET_ALL)
                elif accuracy < 90:
                    print("Evaluation: " +   Fore.GREEN +"Very Good - A" + Style.RESET_ALL)
                else:
                    print("Evaluation: " +   Fore.YELLOW +"Excellent - A+" + Style.RESET_ALL)

                cv2.destroyAllWindows()
                cv2.imshow("Evaluation", eval_painter)
                cv2.imshow(window_name_paint, painter)
                cv2.waitKey(0)
                break
        elif k == ord("e"):#apaga maluco
            print("Key Selected: "+ Fore.YELLOW+"e"+Style.RESET_ALL+"\n\tErasing")
            color = (255,255,255)
        elif k == ord("s"):#ligar e desligar a brush
            brush = False if brush else True
            print("Key Selected: "+ Fore.YELLOW+"s"+Style.RESET_ALL+"\n\tSwitching brush "+(( Fore.GREEN+"ON") if brush else ( Fore.RED+"OFF")) +Style.RESET_ALL)


    # acabar tudo 
    capture.release() #se o programa crashar esta linha n corre e isso acaba com o programa incapaz de inicializar e tem q se desligar a camera do pc senão it dont stop
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()