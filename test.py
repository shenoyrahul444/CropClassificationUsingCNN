"""
import cv2

image_path = "C:\\Users\\shenrahu\\Desktop\\Work\\Personal\\CropClassificationUsingCNN\\1460533143_94615607b7_z.jpg"
# img = cv2.resize(image_path, (100, 50))
img = cv2.imread(image_path)
imcp = img.copy()
print(img.shape)
cv2.imshow('image',img)
cv2.waitKey(500)
cv2.destroyAllWindows()

# cv2.imwrite('output\messigray.png',img)

imrz1= cv2.resize(imcp,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',imrz1)
cv2.waitKey(2000)
cv2.destroyAllWindows()
print(imrz1.shape)

image_output_dimensions = (224, 224)
final_image = cv2.resize(imcp,image_output_dimensions ,interpolation=cv2.INTER_AREA)
cv2.imshow('image',final_image)
cv2.waitKey(2000)
cv2.destroyAllWindows()
print(imrz1.shape)


image_final_location = "{}\{}\{}\{}".format(output_folder_src,category_type,folder_name,file_name)
"""


import folium
#
# m = folium.Map(location=[42.3601,-71.8589],zoom_start = 3)
# tooltip = "Click for more info"
# folium.Marker([42.363600, -71.099500],popup='<strong>Location One</strong>',tooltip=tooltip).add_to(m)
# m.save('map.html')

print(help(folium.Icon))