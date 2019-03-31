import numpy as np
import cv2

#Resize optimization is not fully working
#Currently works with img2 at factor 2, can't find more instances
#I don't know if it is an issue of too low of a resolution

# User input
targetPath = input("Enter the target image: ")
imagePath = input("Enter the image: ")
threshold = input("Enter the threshold: ")
threshold = np.float64(threshold)

# Load images
colorImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
colorTarget = cv2.imread(targetPath, cv2.IMREAD_COLOR)
image = np.float64(cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY))
target = np.float64(cv2.cvtColor(colorTarget, cv2.COLOR_BGR2GRAY))

image /= 255
target /= 255

resize = input("Enter y if you want to optimize: ")
factor = 1;

kRows,kCols = target.shape
rows, cols = image.shape

if resize == "y":
    print("The Refactor is the amount of times it will be reduced")
    print("Inut a value Higher than 1")
    factor = np.float64(input("Enter factor: "))

    if factor > 1:
        kRows = int(kRows/factor)
        kCols = int(kCols/factor)
        rows = int(rows/factor)
        cols = int(cols/factor)
        target = cv2.resize(target,(kCols,kRows))
        image = cv2.resize(image, (cols, rows))
    else:
        print("Invalid Factor")
        factor = 1

# Create matching map

resultArr = np.float64(np.zeros(shape=(rows - kRows,cols - kCols)))
foundRes = []
maxI = rows - kRows
maxJ = cols - kCols
for i in range(0,maxI):
    for j in range(0,maxJ):
        opArr = (image[i:i+kRows,j:j+kCols]).copy()
        resultArr[i,j] = (pow((target - opArr), 2)).sum()
        if resultArr[i,j] <= np.float64(threshold):
            foundRes.append(i)
            foundRes.append(j)

# Print results

cv2.imshow("Image", colorImage)
cv2.waitKey(0)
cv2.imshow("Target", colorTarget)
cv2.waitKey(0)

#Normalize the Matching Map
resultArr = np.uint8((resultArr / np.amax(resultArr))*255)
cv2.imshow("MatchMap", resultArr)
cv2.waitKey(0)

print("Found ", len(foundRes) / 2, " Matching Values")

resultIMG = colorImage.copy()
if len(foundRes) > 0:
    results = np.zeros((40,245,3),np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(results, str(int(len(foundRes) / 2)) + " FOUND", (5,30), font, 1,(0,255,0),2)
    cv2.imshow("Found", results)
    cv2.waitKey(0)

    for i in range(0, int(len(foundRes) / 2)):
        ry1 = int(foundRes[i*2 + 1] * factor)
        rx1 = int(foundRes[i*2] * factor)
        ry2 = int((foundRes[i*2 + 1] + kCols) * factor)
        rx2 = int((foundRes[i*2] + kRows) * factor)
        cv2.rectangle(resultIMG,(ry1, rx1),(ry2,rx2),(0,255,0),2)
else:
    resultIMG = np.zeros((40,245,3),np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resultIMG, "NOTHING FOUND", (5,30), font, 1,(0,255,0),2)

cv2.imshow("Result", resultIMG)
cv2.waitKey(0)

