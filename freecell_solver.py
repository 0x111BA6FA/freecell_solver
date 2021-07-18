import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import statistics
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# card string to text values list
RESTRICTED_VALUES_LIST = ['2','3','4','5','6','7','8','9','1','J','Q','K','A']

def show():
	#cv2.imshow('image', image)

	#plt.figure(1)
	#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	#plt.imshow(cv2.cvtColor(roi_suit, cv2.COLOR_BGR2RGB))
	#plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


#ret, bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# load testing screenshot
image = cv2.imread('f:/work/freecell_solver/freecell.png')

SYM_X = 16
SYM_Y = 15
SYM_X_STEP = 110
SYM_Y_STEP = 30

total_cards_readed = 0
cards_dict = defaultdict(lambda:0)
cards_suits_dict = defaultdict(lambda:0)
readed_cards_list = []

# game stacks
CARD_VALUE_KEY = 'value'
CARD_SUIT_KEY = 'suit'
CARD_BUFFER_KEY = 'buffer'
GOAL_KEY = 'goal'

game = {}
# 4 cards buffer
game[CARD_BUFFER_KEY] = []
# goal of the game
game[GOAL_KEY] = {}
game[GOAL_KEY]['spades'] = []
game[GOAL_KEY]['clubs'] = []
game[GOAL_KEY]['diamonds'] = []
game[GOAL_KEY]['hearts'] = []


for x in range(0, 8):

	# game column
	game[x] = []

	for y in range(0, 7):
		roi_x = 989 + x*SYM_X_STEP
		roi_y = 617 + y*SYM_Y_STEP

		if y >= 4:
			roi_y += 1

		roi = image[roi_y:roi_y+SYM_Y,roi_x:roi_x+SYM_X]

		# detect suit
		roi_suit_x = roi_x + 2
		roi_suit_y = roi_y + SYM_Y - 1
		roi_suit = image[roi_suit_y:roi_suit_y+10,roi_suit_x:roi_suit_x+SYM_X-6]

		bgr = [0, 0, 0]
		pixels_counted = 0

		for k in range(0, len(roi_suit)):
			for j in range(0, len(roi_suit[0])):

				m = np.mean(roi_suit[k][j])

				if m < 230:
					bgr[0] += roi_suit[k][j][0]
					bgr[1] += roi_suit[k][j][1]
					bgr[2] += roi_suit[k][j][2]
					pixels_counted += 1

		# there is no card on that position, go to next card
		if pixels_counted == 0:
			#print('no card detected, skip')
			continue

		#pixels_count = len(roi_suit) * len(roi_suit[0])
		bgr[0] /= pixels_counted
		bgr[1] /= pixels_counted
		bgr[2] /= pixels_counted
		m = np.mean(bgr)

		if bgr[2] > 243:
			suit = 'h'
		elif bgr[2] > 210:
			suit = 'd'
		elif m < 80.0:
			suit = 's'
		else:
			suit = 'c'


		value = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789AKQJ')[0].upper()

		#print( values, suit, x, y, roi_x, roi_y, check_bit, check_bit_x, check_bit_y )
		#print( values, suit, x, y, suit_bit, suit_bit_x, suit_bit_y )
		print( value, suit, x, y, bgr, m )

		#if x == 3 and y == 5:
		#	break

		if value in RESTRICTED_VALUES_LIST:

			#if {'value':value, 'suit':suit} in readed_cards_list:
				#print('<<<<<<<<<<<<<<<<<<<<<<<')

			total_cards_readed += 1
			cards_dict[value] += 1
			cards_suits_dict[suit] += 1
			readed_cards_list += {'value':value, 'suit':suit}

			# translate string to numeric values
			numeric_value = 0
			if value == '1':
				numeric_value = 10
			elif value == 'J':
				numeric_value = 11
			elif value == 'Q':
				numeric_value = 12
			elif value == 'K':
				numeric_value = 13
			elif value == 'A':
				numeric_value = 1
			else:
				numeric_value = int(value)

			# fill game lists
			game[x].append({CARD_VALUE_KEY:numeric_value, CARD_SUIT_KEY:suit})
		else:
			print('failed to recongnize card values {} {}')
			break

	else:
		continue
	break

if total_cards_readed != 52:
	print('readed wrong cards count {}, red_suits {}'.format(total_cards_readed))

for k,v in cards_dict.items():
	if v != 4:
		print('check readed values failed {}, {}'.format(k,v))

for k,v in cards_suits_dict.items():
	if v != 13:
		print('check readed suits failed {}, {}'.format(k,v))


show()
print('ok')
