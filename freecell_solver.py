import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import statistics

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# card string to text nominals list
CARD_NOMINALS_LIST = ['2','3','4','5','6','7','8','9','1','J','Q','K','A']

def show():
	#cv2.imshow('bin', bin)
	#cv2.imshow('gray', gray)
	#cv2.imshow('image', image)
	#cv2.imshow('h', h)

	plt.figure(1)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()

	#plt.figure(2)
	#plt.imshow(cv2.cvtColor(bin, cv2.COLOR_BGR2RGB))
	#plt.show()

	#plt.figure(3)
	#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
	#plt.show()

	#plt.figure(4)
	#plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
	#plt.show()

	#plt.figure(5)
	#plt.imshow(cv2.cvtColor(bin, cv2.COLOR_BGR2RGB))
	#plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def my_thresh():
	for row in range(1, len(image)):
		for col in range(1, len(image[row])):
			blue = int(image[row][col][0])
			green = int(image[row][col][1])
			red = int(image[row][col][2])
			if blue < 220 or green < 220 or red < 220:
				image[row][col][0] = 0
				image[row][col][1] = 0
				image[row][col][2] = 0
				'''
				if blue + green + red < 350:
					image[row][col][0] = 0
					image[row][col][1] = 0
					image[row][col][2] = 0
				'''


#ret, bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# load testing screenshot
image = cv2.imread('f:/work/freecell_solver/freecell.png')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

SYM_X = 16
SYM_Y = 15
SYM_X_STEP = 110
SYM_Y_STEP = 30

cards_readed = 0

for x in range(0, 8):
	for y in range(0, 7):
		roi_x = 989 + x*SYM_X_STEP
		roi_y = 617 + y*SYM_Y_STEP

		if y >= 4:
			roi_y += 1

		roi = image[roi_y:roi_y+SYM_Y,roi_x:roi_x+SYM_X]
		#gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		#ret, bin = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

		# check if there is card, check if bit is green
		check_bit_x = roi_x + SYM_X - 1
		check_bit_y = roi_y + 80
		check_bit = image[check_bit_y, check_bit_x]
		if check_bit[1] > check_bit[0] and check_bit[1] > check_bit[2] and statistics.mean(check_bit) < 200:
			print('no card detected, skipped', check_bit, check_bit_x, check_bit_y)
			continue

		nominal = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789AKQJ')[0].upper()
		print( nominal, x, y, roi_x, roi_y, check_bit, check_bit_x, check_bit_y )
		if nominal not in CARD_NOMINALS_LIST:
			print('failed to recongnize card nominal {} {}')
			break
		else:
			cards_readed += 1
	else:
		continue
	break

if cards_readed != 52:
	print('readed wrong cards count {}'.format(cards_readed))
else:
	print('readed ok')

#print(gray)

#my_thresh()



#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#h,s,v = cv2.split(hsv)
#h[h<150]=0
#h[h>180]=0

# check zoomed letter
# downscale

show()
exit(0)

'''
# remove green noise
for row in range(1, len(image)):
	for col in range(1, len(image[row])):
		blue = image[row][col][0]
		green = image[row][col][1]
		red = image[row][col][2]
		if green > max(blue, red) + 20:
			image[row][col][0]  = 0
			image[row][col][1] = 0
			image[row][col][2] = 0
'''

#ret, bin = cv2.threshold(gray, 150, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

# no need blur?
#gray = cv2.GaussianBlur(gray, (3, 3), 0)
#edges = cv2.Canny(img, 60, 255)

# getting gray channel
#gray = cv2.split(img)

#roi = img[0:1,0:1]

#print( 'rows', len(gray[0]) )
#print( 'columns', len(gray[0][0]) )
#print( gray )
#img1 = np.zeros((200,200,1), dtype=np.uint8)

canny = cv2.Canny(bin, 1, 1)


# Find my contours
window_contours = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
cntrRect = []
for countour in window_contours:
	epsilon = 0.05*cv2.arcLength(countour,True)
	approx = cv2.approxPolyDP(countour,epsilon,True)
	print('len approx {}'.format(len(approx)))
	if len(approx) == 4:
		#cv2.drawContours(roi,cntrRect,-1,(0,255,0),2)
		#cv2.imshow('Roi Rect ONLY',roi)
		cv2.drawContours(image, countour, -1, (0, 255, 0), 2)
		#cv2.imshow('Roi Rect ONLY', img)
		cntrRect.append(approx)


# detect window

# read game
# read cards by top left 2 symbols

# solve game
# check random moves as stack of moves

# execute solution


print('ok')
