import numpy, cv2, operator, sudoku, time

from keras.models import load_model
from keras.models import model_from_json

classifier = load_model("./classifier.h5")

EPS = 4
STEP = 28 + 2 * EPS
SIDE = 9 * STEP
FRAME_RATE = 10

capture, flag = cv2.VideoCapture(0), 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('outputput.avi', fourcc, 30.0, (1080, 620))
prev = 0

while True:
  time_elapsed = time.time() - prev
  ret, frame = capture.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (7, 7), 0)
  
  thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
  bigContour, maxArea = None, 0

  for c in contours:
    area = cv2.contourArea(c)
    if area > 80000:
      perimeter = cv2.arcLength(c, True)
      polygon = cv2.approxPolyDP(c, 0.01 * perimeter, True)
      if area > maxArea and len(polygon) == 4:
        maxArea, bigContour = area, polygon

  if bigContour is not None:
    cv2.drawContours(frame, [bigContour], 0, (0, 255, 0), 2)
    points = numpy.vstack(bigContour).squeeze()
    points = sorted(points, key = operator.itemgetter(1))
    
    if points[0][0] > points[1][0]: points[0], points[1] = points[1], points[0]
    if points[2][0] > points[3][0]: points[2], points[3] = points[3], points[2]

    from_list = numpy.float32([points[0], points[1], points[2], points[3]])
    to_list = numpy.float32([[0, 0], [SIDE, 0], [0, SIDE], [SIDE, SIDE]])

    transform = cv2.getPerspectiveTransform(from_list, to_list)

    puzzle = cv2.warpPerspective(frame, transform, (SIDE, SIDE))
    puzzle = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
    puzzle = cv2.adaptiveThreshold(puzzle, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)
    
    cv2.imshow('Puzzle', puzzle)

    if flag == 0:
      result = None
      extract_grid, saved_grid = [], []

      for y in range(9):
        row = ''
        for x in range(9):
          y_low = y * STEP + EPS
          y_high = (y + 1) * STEP - EPS
          x_low = x * STEP + EPS
          x_high = (x + 1) * STEP - EPS
          
          cell = puzzle[y_low : y_high, x_low : x_high]
          cv2.imwrite('Cell' + str(y) + str(x) + ".png", cell)
          cell = cell.reshape(1, 28, 28, 1)
          
          if cell.sum() > 7000:
            prediction = numpy.argmax(classifier.predict(cell), axis = -1)
            row += str(prediction[0])
          else:
            row += '0'

        extract_grid.append(row)

      if extract_grid != saved_grid:
        saved_grid = extract_grid.copy()
        print(saved_grid)
        result = sudoku.sudoku(saved_grid)

    print('Solution:', result)

    if result is not None:
      flag, fond = 1, numpy.zeros(shape = (SIDE, SIDE, 3), dtype = numpy.float32)
      
      for y in range(len(result)):
        for x in range(len(result[y])):
          if saved_grid[y][x] == '0':
            cv2.putText(fond, str(result[y][x]), (x * STEP + EPS + 3, (y + 1) * STEP - EPS - 3), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 0, 255), 1)
      
      transform = cv2.getPerspectiveTransform(to_list, from_list)
      h, w, c = frame.shape
      fondP = cv2.warpPerspective(fond, transform, (w, h))
      img2gray = cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
      ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
      mask = mask.astype('uint8')
      mask_inv = cv2.bitwise_not(mask)
      img1_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)
      img2_fg = cv2.bitwise_and(fondP, fondP, mask = mask).astype('uint8')
      dst = cv2.add(img1_bg, img2_fg)
      dst = cv2.resize(dst, (1080, 620))
      cv2.imshow('Frame', dst)
      output.write(dst)

    else:
      frame = cv2.resize(frame, (1080, 620))
      cv2.imshow('Frame', frame)
      output.write(frame)
  else:
    flag = 0
    frame = cv2.resize(frame, (1080, 620))
    cv2.imshow('Frame', frame)
    output.write(frame)

  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'): break	


output.release()
cap.release()
cv2.destroyAllWindows()
