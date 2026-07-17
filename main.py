# cv2 is the installation of Computer Vision that is used for image processing, video analysis and machine learning
import cv2
# numpy as np imports the entire numpy library and then it creates np as a nickname
import numpy as np



# Created a function called load_image with a paramter called path 
def load_image(path):

# Created a variable called img
    # img holds the value of the library call of cv2, which uses computer vision
    # .imread is a function that is used in Python to read a image file into storage, which allows numpy to convert it into a numerical array so it can use it
    # Then the parameter called path is used as the image
    img = cv2.imread(path)

# If statement to check if there is no image uploaded it will print an error message and return None
    if img is None:
        print("Error: Image file not found!")
        return None
    # No matter what happens it will return img
    return img

# Function called process_images that converts the img into black n white and blurs it out
# Gray scale, Gaussian Blur, and Canny
def process_edges(color_image):

    # Variable that changes the color of the image to gray
    gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)


    # Variable called blurred that equals cv2.GaussianBlur, which blurs the uncessasary details like the floor background or textures
    
    # Adjust the blur from (5,5) to (9,9) to ensure that the wood grains are blended in
    blurred = cv2.GaussianBlur(gray,(9,9),0)

    # Variable called edges that uses .Canny, which allows the edges by determing the major brightness difference

    # Lowering the .Canny measurements to make sure that the foot is still in tact
    edges = cv2.Canny(blurred,80,180)

    
    # This creates a small 3x3 digital brush that can be used to connect the gaps
    kernel = np.ones((5,5),np.uint8)

    # This brush blurs the unessential things and connects the lines of the essential parts of the image
    closed_edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)

    return closed_edges

#def close_gaps_in_edges(edge_image):
    # Variable that is called kernal that uses the np(which is the numpy import)
    # This creates a 3x3 matrix to act like a brush
    ##




def find_paper_corners(edge_image):
    # contours is the variable that is going to hold a list of all the mathematicals from inside the variable
        # It will provide them in x,y coordinates
        # With the use of "findContours", it provides us the contours and the hierachy, so we implement the _ to tell the code to only use the contours and leave out the hierachy
        # .findContours is from cv2 that analyzes the black and white image and traces the white
        # .RETR_EXTERNAL provide strict instructions to only trace the outer most boundary of the paper while forgetting what would be inside the paper
        # .CHAIN_APPROX_SIMPLE provides mememory and processing optimization meaning that if a shape had 1000 pixels, it would only grab the essential ones such as the endpoints to reduce the storage
    contours, _ = cv2.findContours(edge_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Sorted() sorts the list from smallest to largest
    # With the (reverse=True) flips it so it goes from largest to smallest
    # Contours is a list of all of the continous line outputs such as the foot, paper, and all the lines
    # Instead of sorting by alphabetically or numerically, it takes the value of the surface area 
    sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)



# Iteration for the list of contours
    for contour in contours:


        # This skips the smaller small shapes and only gets the bigger ones
        if cv2.contourArea(contour) < 5000:
            continue
        # This variable holds the value of the perimeter
        # .arcLength finds the perimeter of the shape
        # contour is the shape itself being analyzed
        # True tells the code that is a continous shape that is connected without any breaks
        perimeter = cv2.arcLength(contour,True)

        # .approxPolyDP optimizes the pixels by throwing away the unecessary lines by only getting the end points
        # contour is the shape
        # 0.02 * perimeter is the brains of the code, this allows the code to be adaptable by using 2% of the shape's total size. Basically no matter the distant from where the picture is taken, the math will automatically adapt to the size of object in the image
        # True provides the same as before, it ensures that it is a closed loop meaning that it will connect perfectly
        approx_shape = cv2.approxPolyDP(contour,0.02 * perimeter,True)


        
    # Check statement that verifies if shape has 4 corners
        if len(approx_shape) == 4:
            print(f"[Found] Paper detected with 4 coordinates!")
            return approx_shape
    print(f"[Error] Could not find 4-cornered shape for calibration")
    return None


# Main Function

if __name__ == "__main__":

    # Variable called raw_image that runs the load_image function with the value of test_foot.jpg in it
    raw_image = load_image("test_foot.jpg")

    
    # Checks if raw_image is not None, which means if it has a value, continue
    if raw_image is not None:


        # Extract edges
        # It runs the process_edges function on the raw_image
        detected_edges = process_edges(raw_image)

        # Render
        # .imshow shows the image into a 2D array or pixels
        cv2.imshow("Detected Edges: ",detected_edges)


        # Variable that holds the return value for of find_paper_corners function. The detected_edges is the completed black and white photo
        paper_coordinates = find_paper_corners(detected_edges)


    # Checks if paper_coordinates is not None to then run and give the paper_coordinates
    # Only proceeds if there is a 4 cornered shape
        if paper_coordinates is not None:
            print("Paper Corner Coordinates: \n", paper_coordinates)



        # .imshow shows the detected images that got repaired from the 3x3 brush
        cv2.imshow("Healed Edges: ",detected_edges)
        # Keep open until until told so
        cv2.waitKey(0)
        cv2.destroyAllWindows()
