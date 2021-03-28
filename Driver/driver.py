from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator

def preprocess(img,skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)  
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    if skip_dilate==False:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc

def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def findsudoku(img):
  newimg = preprocess(img.copy(),skip_dilate=True)
  contours, _ = cv2.findContours(newimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  #showcontours = cv2.drawContours(img, [contours[0]], 0, (0,128,0), 3)
  polygon = contours[0]
  bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
  top_left, top_right, bottom_right, bottom_left = polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]
  src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
  side = max([  distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right) ])
  dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
  m = cv2.getPerspectiveTransform(src, dst)
  finalgrid=cv2.warpPerspective(img, m, (int(side), int(side)))
  return finalgrid



def makefinalgrid(grid):
    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9
    ret,grid = cv2.threshold(grid,127,255,cv2.THRESH_BINARY)

    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])
    for i in range(9):
        for j in range(9):
          finalgrid[i][j] = np.array(finalgrid[i][j])
    try:
        for i in range(9):
            for j in range(9):
                os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])
    for i in range(9):
      for j in range(9):
        finalgrid[i][j]=cv2.resize(finalgrid[i][j],(28,28))
    return finalgrid

def makesudoku(finalgrid):
    sudoku = np.zeros([9,9])
    for i in range(9):
      for j in range(9):
        if(finalgrid[i][j].sum()>37500):
          sudoku[i][j]=getnumber((finalgrid[i][j]))
        else:
          sudoku[i][j]=0
    return sudoku

def getnumber(img):
  plt.imshow(img)
  resize = cv2.resize(img.copy(),(28,28))
  reshaped = resize.reshape(1,28,28,1)
  loaded_model_pred = loaded_model.predict(reshaped , verbose = 0)[0]
  return np.argmax(loaded_model_pred)+1
def isvalid(sudoku,i,j,po):
  for jj in range(9):
    if sudoku[i][jj]==po:
      return False
  for ii in range(9):
    if sudoku[ii][j]==po:
      return False
  smi = (i//3)*3
  smj = (j//3)*3
  #print(smi,smj)
  for x in range(3):
    for y in range(3):
      if (sudoku[smi+x][smj+y]==po):
        return False
  return True
    
sodfinal = []

def solvesudoku(sudoku,i,j):
  if (i==9):
    sodfinal = sudoku
    return

  if j==8:
    ni=i+1
    nj=0
  else:
    ni=i
    nj=j+1
  
  if (sudoku[i][j]!=0):
    solvesudoku(sudoku,ni,nj)
  else:
    for po in range(10):
      if (isvalid(sudoku,i,j,po)==True):
        sudoku[i][j]=po
        solvesudoku(sudoku,ni,nj)
      sudoku[i][j]=0

def Solve_Sudoku(imgpath):
    img=cv2.imread(imgpath)
    finalimg = findsudoku(img)
    finalimg = cv2.cvtColor(finalimg, cv2.COLOR_BGR2GRAY)
    finalimg = cv2.bitwise_not(cv2.adaptiveThreshold(finalimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)) 
    loaded_model = load_model('./test_model')
    finalgrid = makefinalgrid(finalimg)
    sudoku = makesudoku(finalgrid)
    solvesudoku(sudoku)
    print(sodfinal)
