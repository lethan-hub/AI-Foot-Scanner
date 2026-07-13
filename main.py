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

    return edges

def close_gaps_in_edges(edge_image):
    # Variable that is called kernal that uses the np(which is the numpy import)
    # This creates a 3x3 matrix to act like a brush
    
    kernel = np.ones((3,3),np.uint8)

    # This brush blurs the unessential things and connects the lines of the essential parts of the image
    closed_edges = cv2.morphologyEx(edge_image,cv2.MORPH_CLOSE,kernel)

    return closed_edges


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


        # Keep open until until told so
        cv2.waitKey(0)
        cv2.destroyAllWindows()
