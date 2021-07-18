import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import statistics
from collections import defaultdict
import json

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# dumped game
GAME_DUMP_FILE = '{"field": [[{"value": 3, "suit": "s", "string": "3s"}, {"value": 7, "suit": "h", "string": "7h"}, {"value": 9, "suit": "h", "string": "9h"}, {"value": 3, "suit": "d", "string": "3d"}, {"value": 13, "suit": "d", "string": "Kd"}, {"value": 2, "suit": "d", "string": "2d"}, {"value": 4, "suit": "s", "string": "4s"}], [{"value": 2, "suit": "s", "string": "2s"}, {"value": 12, "suit": "s", "string": "Qs"}, {"value": 12, "suit": "h", "string": "Qh"}, {"value": 8, "suit": "h", "string": "8h"}, {"value": 12, "suit": "c", "string": "Qc"}, {"value": 13, "suit": "c", "string": "Kc"}, {"value": 4, "suit": "c", "string": "4c"}], [{"value": 7, "suit": "s", "string": "7s"}, {"value": 10, "suit": "s", "string": "10s"}, {"value": 8, "suit": "c", "string": "8c"}, {"value": 9, "suit": "c", "string": "9c"}, {"value": 4, "suit": "h", "string": "4h"}, {"value": 13, "suit": "h", "string": "Kh"}, {"value": 11, "suit": "c", "string": "Jc"}], [{"value": 10, "suit": "s", "string": "10s"}, {"value": 8, "suit": "s", "string": "8s"}, {"value": 1, "suit": "c", "string": "Ac"}, {"value": 4, "suit": "d", "string": "4d"}, {"value": 5, "suit": "s", "string": "5s"}, {"value": 11, "suit": "h", "string": "Jh"}, {"value": 1, "suit": "s", "string": "As"}], [{"value": 6, "suit": "d", "string": "6d"}, {"value": 6, "suit": "h", "string": "6h"}, {"value": 10, "suit": "h", "string": "10h"}, {"value": 11, "suit": "c", "string": "Jc"}, {"value": 5, "suit": "d", "string": "5d"}, {"value": 8, "suit": "d", "string": "8d"}], [{"value": 9, "suit": "c", "string": "9c"}, {"value": 6, "suit": "s", "string": "6s"}, {"value": 5, "suit": "c", "string": "5c"}, {"value": 1, "suit": "h", "string": "Ah"}, {"value": 12, "suit": "d", "string": "Qd"}, {"value": 13, "suit": "s", "string": "Ks"}], [{"value": 2, "suit": "c", "string": "2c"}, {"value": 2, "suit": "h", "string": "2h"}, {"value": 9, "suit": "d", "string": "9d"}, {"value": 5, "suit": "h", "string": "5h"}, {"value": 11, "suit": "d", "string": "Jd"}, {"value": 3, "suit": "h", "string": "3h"}], [{"value": 1, "suit": "d", "string": "Ad"}, {"value": 3, "suit": "c", "string": "3c"}, {"value": 10, "suit": "d", "string": "10d"}, {"value": 7, "suit": "d", "string": "7d"}, {"value": 6, "suit": "s", "string": "6s"}, {"value": 7, "suit": "c", "string": "7c"}]], "buffer": [], "goal": {"spades": 0, "clubs": 0, "diamonds": 0, "hearts": 0}}'

# card string to text values list
RESTRICTED_VALUES_LIST = ['2','3','4','5','6','7','8','9','1','J','Q','K','A']

SYM_X = 16
SYM_Y = 15
SYM_X_STEP = 110
SYM_Y_STEP = 30

GAME_COLUMNS_COUNT = 8

# game stacks
CARD_VALUE_KEY = 'value'
CARD_SUIT_KEY = 'suit'
CARD_STRING_KEY = 'string'
CARD_BUFFER_KEY = 'buffer'
GOAL_KEY = 'goal'
FIELD_KEY = 'field'


# game structure
game = {}
# game field
game[FIELD_KEY] = []
# 4 cards buffer
game[CARD_BUFFER_KEY] = []
# goal of the game, 0 - none, 1 - A, .. 13 - K
game[GOAL_KEY] = {}
game[GOAL_KEY]['spades'] = 0
game[GOAL_KEY]['clubs'] = 0
game[GOAL_KEY]['diamonds'] = 0
game[GOAL_KEY]['hearts'] = 0

def show():
	#cv2.imshow('image', image)

	#plt.figure(1)
	#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	#plt.imshow(cv2.cvtColor(roi_suit, cv2.COLOR_BGR2RGB))
	#plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def load_image():

	global game

	# <<<<<<<<<<<<<<<<<<<<< load dumped game
	game = json.loads(GAME_DUMP_FILE)
	print('game loaded')
	return


    #ret, bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

	# load testing screenshot
	image = cv2.imread('f:/work/freecell_solver/freecell.png')

	total_cards_readed = 0
	cards_dict = defaultdict(lambda:0)
	cards_suits_dict = defaultdict(lambda:0)
	readed_cards_list = []

	for x in range(0, 8):

		# append game column stack
		game[FIELD_KEY].append([])

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

			# pixels on suit area are white - there is no card - go to next
			if pixels_counted < 10:
				#print('no card detected, skip')
				continue

			bgr[0] /= pixels_counted
			bgr[1] /= pixels_counted
			bgr[2] /= pixels_counted
			m = np.mean(bgr)

			# decide suit
			# hearts are many red pixels
			if bgr[2] > 243:
				suit = 'h'
			# diamonds are red, but lesser than  hearts
			elif bgr[2] > 210:
				suit = 'd'
			# spades are more black than clubs
			elif m < 80.0:
				suit = 's'
			else:
				suit = 'c'

			# translate value of card to text
			value = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789AKQJ')[0].upper()

			#print( values, suit, x, y, roi_x, roi_y, check_bit, check_bit_x, check_bit_y )
			#print( values, suit, x, y, suit_bit, suit_bit_x, suit_bit_y )
			#print( value, suit, x, y, bgr, m )

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
					value = '10'
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
				game[FIELD_KEY][x].append({CARD_VALUE_KEY:numeric_value, CARD_SUIT_KEY:suit, CARD_STRING_KEY:'{}{}'.format(value, suit) })
			else:
				print('failed to recongnize card values {} {}')
				break

		else:
			continue
		break

	# result cheks

	# loaded exactly 52 cards
	if total_cards_readed != 52:
		print('readed wrong cards count {}, red_suits {}'.format(total_cards_readed))

	# every card loaded 4 times
	for k,v in cards_dict.items():
		if v != 4:
			print('check readed values failed {}, {}'.format(k,v))

	# every suit has 4 cards
	for k,v in cards_suits_dict.items():
		if v != 13:
			print('check readed suits failed {}, {}'.format(k,v))

	# unique card check?

	# dump oaded game
	file = open('./game.dump', 'w')
	file.write(json.dumps(game))
	print('game dumped')

	print('game loaded')


def print_current_field():

	global game

	# buffer state
	buffer_string = ''

	for i in range(0,4):
		if len(buffer_string): buffer_string += '\t'
		if len(game[CARD_BUFFER_KEY]) > i:
			buffer_string += game[CARD_BUFFER_KEY][i]
		else:
			buffer_string += '__'

	# result state
	result_string = '\t{}\t{}\t{}\t{}'.format(game[GOAL_KEY]['hearts'], game[GOAL_KEY]['clubs'], game[GOAL_KEY]['diamonds'], game[GOAL_KEY]['spades'])

	print('\t\t\t    |\t\t\t\t')
	#print('1\t2\t3\t4   |\t5\t6\t7\t8')
	print('{}  |{}'.format(buffer_string, result_string))
	print('\t\t\t    |\t\t\t\t\n')

	# print field
	row = 0
	while True:

		display_flag = False

		row_string = ''

		for col in range(0, GAME_COLUMNS_COUNT):

			if len(game[FIELD_KEY][col]) > row:
				row_string += '{}\t'.format( game[FIELD_KEY][col][row][CARD_STRING_KEY] )

		if len(row_string) == 0: break

		print( row_string )
		row += 1

def solve_game():

	global game

	#print( 'solve' )
	#print( game[FIELD_KEY][0][3][CARD_VALUE_KEY], game[FIELD_KEY][0][3][CARD_SUIT_KEY] )

	# main solving loop
	while True:

		row_string = ''

		for col in range(0, GAME_COLUMNS_COUNT):
			row_string += '{}\t'.format( game[FIELD_KEY][col][-1][CARD_STRING_KEY] )


		break


load_image()
#solve_game()
print_current_field()

#show()
print('ok')
